#!/usr/bin/env python3
"""
Multi-LLM Topics - Streamlit Interface

A lightweight GUI for ensemble topic discovery with multiple LLMs.

Usage:
    streamlit run app.py
"""

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(
    page_title="Multi-LLM Topics",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .topic-tag {
        display: inline-block;
        background: #f0f2f6;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 2px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Default recommended models (used if API fetch fails)
DEFAULT_DISCOVERY_MODELS = [
    "google/gemini-2.0-flash-001",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-nano",
    "mistralai/ministral-8b-2412",
    "deepseek/deepseek-chat-v3-0324",
]

DEFAULT_CONSOLIDATION_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
    "google/gemini-2.0-flash-001",
]

DEFAULT_ASSIGNMENT_MODELS = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-haiku-4.5",
]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_openrouter_models():
    """Fetch all available models from OpenRouter API with pricing."""
    try:
        api_key = st.session_state.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=10
        )
        if response.status_code == 200:
            models = response.json().get("data", [])
            # Build dict with pricing info
            model_info = {}
            for m in models:
                model_id = m.get("id")
                if model_id:
                    pricing = m.get("pricing", {})
                    try:
                        # Pricing is per token, convert to per 1M tokens
                        prompt_cost = float(pricing.get("prompt", 0)) * 1_000_000
                        completion_cost = float(pricing.get("completion", 0)) * 1_000_000
                    except (ValueError, TypeError):
                        prompt_cost = 0
                        completion_cost = 0
                    model_info[model_id] = {
                        "prompt_cost": prompt_cost,  # $ per 1M tokens
                        "completion_cost": completion_cost,
                        "context_length": m.get("context_length", 0),
                    }
            return model_info
    except Exception as e:
        st.warning(f"Could not fetch models from OpenRouter: {e}")
    return {}


def get_model_cost_estimate(model_id: str, n_docs: int, task: str = "discovery") -> float:
    """Estimate cost for a task based on model pricing."""
    models = fetch_openrouter_models()
    if model_id not in models:
        return 0.0

    pricing = models[model_id]
    prompt_cost = pricing["prompt_cost"]  # per 1M tokens
    completion_cost = pricing["completion_cost"]

    # Rough token estimates per document
    if task == "discovery":
        # ~500 tokens prompt + ~50 tokens response per doc
        tokens_in = n_docs * 500
        tokens_out = n_docs * 50
    elif task == "consolidation":
        # One big call: ~20 tokens per topic input + ~50 tokens per topic output
        tokens_in = n_docs * 20  # n_docs here is n_topics
        tokens_out = n_docs * 50
    else:  # assignment
        # ~1500 tokens prompt (taxonomy) + ~100 tokens response per doc
        tokens_in = n_docs * 1500
        tokens_out = n_docs * 100

    cost = (tokens_in / 1_000_000) * prompt_cost + (tokens_out / 1_000_000) * completion_cost
    return cost


def format_cost(cost: float) -> str:
    """Format cost as readable string."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def get_models_by_category():
    """Get models organized for different use cases."""
    all_models_info = fetch_openrouter_models()
    all_models = sorted(all_models_info.keys())

    if not all_models:
        return {
            "all": DEFAULT_DISCOVERY_MODELS + DEFAULT_CONSOLIDATION_MODELS + DEFAULT_ASSIGNMENT_MODELS,
            "discovery": DEFAULT_DISCOVERY_MODELS,
            "consolidation": DEFAULT_CONSOLIDATION_MODELS,
            "assignment": DEFAULT_ASSIGNMENT_MODELS,
            "pricing": {},
        }

    # Categorize models
    fast_models = [m for m in all_models if any(x in m.lower() for x in ["flash", "haiku", "mini", "nano", "lite", "8b", "7b"])]
    strong_models = [m for m in all_models if any(x in m.lower() for x in ["sonnet", "opus", "gpt-4", "claude-3", "gemini-pro", "gemini-2"])]

    return {
        "all": all_models,
        "discovery": fast_models if fast_models else all_models[:20],
        "consolidation": strong_models if strong_models else all_models[:10],
        "assignment": fast_models if fast_models else all_models[:20],
        "pricing": all_models_info,
    }

# Default Prompts
DEFAULT_DISCOVERY_PROMPT = """Identify the main topic of this document.

[Existing Topics]
{topics}

[Instructions]
1. Read the document carefully and identify its PRIMARY topic - the main theme or subject.
2. Topic labels should be descriptive and generalizable (2-4 words).
3. If an existing topic fits well, use it. Otherwise create a new one.
4. Use lowercase_with_underscores format.

Good topic examples: "privacy_boundaries", "relationship_communication", "trust_issues", "financial_disagreements", "family_interference", "commitment_concerns", "emotional_distance", "work_life_balance"

Bad topic examples: "number_1", "post_about_thing", "miscellaneous", "other"

[Document]
{post}

Respond with JSON:
{{"action": "existing" or "new", "topic": "descriptive_topic_label", "description": "One sentence description (required for new topics)"}}"""

DEFAULT_CONSOLIDATION_PROMPT = """Consolidate these topic labels into a coherent taxonomy.

These topics were discovered by different LLMs. Due to different naming conventions, there is overlap. Your task:
1. Merge topics that represent the SAME concept (different wording for same idea)
2. Keep topics SEPARATE if they represent meaningfully different concepts
3. Aim for 50-80 final topics

Map EVERY topic to a category. Provide complete lists, not examples.

RAW TOPICS ({n_topics} total):
{topics}

Respond with JSON:
{{
  "taxonomy": [
    {{"topic": "topic_label", "description": "What this covers", "source_topics": ["all", "matching", "topics"]}}
  ],
  "unmapped": []
}}"""

DEFAULT_ASSIGNMENT_PROMPT = """Assign 1-3 topic labels to this document.

Rules:
- Select the PRIMARY topic (most relevant) first
- Add 1-2 SECONDARY topics only if clearly applicable
- Focus on core subject matter, not minor details

TAXONOMY ({n_topics} topics):
{taxonomy}

DOCUMENT:
{post}

Respond with JSON only:
{{"primary_topic": "most_relevant", "secondary_topics": ["other1", "other2"], "reasoning": "Brief explanation"}}"""

# Initialize prompts in session state
if "discovery_prompt" not in st.session_state:
    st.session_state["discovery_prompt"] = DEFAULT_DISCOVERY_PROMPT
if "consolidation_prompt" not in st.session_state:
    st.session_state["consolidation_prompt"] = DEFAULT_CONSOLIDATION_PROMPT
if "assignment_prompt" not in st.session_state:
    st.session_state["assignment_prompt"] = DEFAULT_ASSIGNMENT_PROMPT


def get_client():
    """Get OpenAI client configured for OpenRouter."""
    api_key = st.session_state.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def process_single_post(client, model, post_text, topics, lock):
    """Process a single post for topic discovery."""
    try:
        # Handle None or non-string values
        if post_text is None or pd.isna(post_text):
            return "", False
        post_text = str(post_text)
        if len(post_text.strip()) == 0:
            return "", False

        topics_formatted = "\n".join(f"- {t}: {d.get('description', '')}"
                                      for t, d in topics.items()) if topics else "(No topics yet)"

        prompt = st.session_state["discovery_prompt"].format(topics=topics_formatted, post=post_text[:6000])
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            return "", False

        parsed = json.loads(content.strip())
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}

        topic = parsed.get("topic", "")
        if isinstance(topic, str):
            topic = re.sub(r'[^a-z0-9_]', '_', topic.lower().strip())
            topic = re.sub(r'_+', '_', topic).strip('_')

        if not topic:
            return "", False

        action = parsed.get("action", "existing")
        description = parsed.get("description", "")

        with lock:
            if action == "new" and topic not in topics:
                topics[topic] = {"description": description, "count": 1}
                return topic, True
            elif topic in topics:
                topics[topic]["count"] = topics[topic].get("count", 0) + 1
                return topic, False
            else:
                topics[topic] = {"description": description, "count": 1}
                return topic, True

    except Exception:
        return "", False


def run_discovery(client, model, texts, n_samples, progress_bar, status_text,
                  batch_size=50, early_stop_batches=3):
    """Run topic discovery on texts with early stopping."""
    sample = texts[:n_samples] if len(texts) > n_samples else texts
    topics = {}
    lock = threading.Lock()

    total = len(sample)
    completed = 0
    new_topics_total = 0
    batches_without_new = 0
    early_stopped = False

    # Process in batches for early stopping
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_texts = sample[batch_start:batch_end]
        new_in_batch = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_single_post, client, model, text, topics, lock): i
                       for i, text in enumerate(batch_texts)}

            for future in as_completed(futures):
                topic, is_new = future.result()
                completed += 1
                if is_new:
                    new_topics_total += 1
                    new_in_batch += 1

                progress_bar.progress(completed / total)
                status_text.text(f"Processed {completed}/{total} documents | {len(topics)} topics ({new_topics_total} new)")

        # Check early stopping
        if new_in_batch == 0:
            batches_without_new += 1
            if batches_without_new >= early_stop_batches:
                early_stopped = True
                status_text.text(f"Early stop: {batches_without_new} batches without new topics. {len(topics)} topics found.")
                break
        else:
            batches_without_new = 0

    if not early_stopped:
        status_text.text(f"Complete: {len(topics)} topics from {completed} documents")

    return topics


def run_consolidation(client, model, topics, progress_bar, status_text):
    """Consolidate topics into taxonomy."""
    topics_formatted = "\n".join(f"- {t}" for t in sorted(topics))
    prompt = st.session_state["consolidation_prompt"].format(n_topics=len(topics), topics=topics_formatted)

    status_text.text(f"Sending {len(topics)} topics to {model}...")
    progress_bar.progress(0.3)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    progress_bar.progress(0.8)
    content = response.choices[0].message.content

    if not content:
        return {"taxonomy": [], "unmapped": []}

    # Strip markdown if present
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean[clean.find("\n") + 1:]
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    progress_bar.progress(1.0)
    status_text.text("Consolidation complete!")

    return json.loads(clean)


def run_assignment(client, model, texts, ids, taxonomy, progress_bar, status_text):
    """Assign topics to documents."""
    taxonomy_formatted = "\n".join(f"- {t['topic']}: {t.get('description', '')}" for t in taxonomy)
    valid_labels = {t['topic'] for t in taxonomy}

    results = []
    total = len(texts)

    def assign_single(text, doc_id):
        prompt = st.session_state["assignment_prompt"].format(
            n_topics=len(taxonomy),
            taxonomy=taxonomy_formatted,
            post=text[:8000]
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                return {"id": doc_id, "error": "empty_response"}

            result = json.loads(content.strip())
            primary = result.get("primary_topic", "")
            secondary = result.get("secondary_topics", [])

            # Validate
            if primary not in valid_labels:
                primary_norm = primary.lower().replace(" ", "_")
                matches = [l for l in valid_labels if primary_norm in l.lower()]
                primary = matches[0] if matches else "unknown"

            valid_secondary = [t for t in secondary if t in valid_labels]

            return {
                "id": doc_id,
                "primary_topic": primary,
                "secondary_topics": valid_secondary,
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            return {"id": doc_id, "error": str(e)}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(assign_single, text, doc_id): i
                   for i, (text, doc_id) in enumerate(zip(texts, ids))}

        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Assigned {i + 1}/{total} documents")

    return results


# Sidebar
with st.sidebar:
    st.title("🏷️ Multi-LLM Topics")
    st.caption("Ensemble topic discovery")

    st.divider()

    # API Key
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=st.session_state.get("api_key", os.environ.get("OPENROUTER_API_KEY", "")),
        help="Get your key at [openrouter.ai](https://openrouter.ai)"
    )
    if api_key:
        # Only validate if key changed
        if api_key != st.session_state.get("api_key_validated", ""):
            with st.spinner("Validating API key..."):
                try:
                    response = requests.get(
                        "https://openrouter.ai/api/v1/auth/key",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.session_state["api_key"] = api_key
                        st.session_state["api_key_validated"] = api_key
                        st.session_state["api_key_valid"] = True
                        data = response.json().get("data", {})
                        # Show credit balance if available
                        usage = data.get("usage", 0)
                        limit = data.get("limit")
                        if limit:
                            st.success(f"API key valid (${usage:.2f} / ${limit:.2f} used)")
                        else:
                            st.success("API key valid")
                    else:
                        st.session_state["api_key_valid"] = False
                        st.error("Invalid API key")
                except Exception as e:
                    st.session_state["api_key_valid"] = False
                    st.error(f"Could not validate key: {e}")
        elif st.session_state.get("api_key_valid"):
            st.success("API key valid")

    st.divider()

    # Sample data option
    if "data" not in st.session_state:
        if st.button("📋 Load Sample Data", help="Load 100 sample posts to try out the app"):
            sample_path = Path(__file__).parent / "sample_data.csv"
            if sample_path.exists():
                df = pd.read_csv(sample_path)
                st.session_state["data"] = df
                st.session_state["text_column"] = "text"
                st.session_state["id_column"] = "id"
                st.rerun()
            else:
                st.error("Sample data file not found")
        st.caption("or upload your own:")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Data",
        type=["csv", "parquet"],
        help="CSV or Parquet file with text data. The app auto-detects common text columns (text, body, selftext, content) but you can change it after uploading."
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)
        st.session_state["data"] = df

    # Show data info and column selection when data is loaded
    if "data" in st.session_state:
        df = st.session_state["data"]
        st.success(f"Loaded {len(df):,} rows")

        # Auto-detect text column (common names)
        text_col_candidates = ["text", "body", "selftext", "content", "message", "post", "document"]
        columns = df.columns.tolist()
        default_text_idx = 0
        for candidate in text_col_candidates:
            if candidate in columns:
                default_text_idx = columns.index(candidate)
                break

        # Auto-detect ID column
        id_col_candidates = ["id", "post_id", "doc_id", "index", "name"]
        default_id = "(row index)"
        for candidate in id_col_candidates:
            if candidate in columns:
                default_id = candidate
                break

        # Use existing selection if available, otherwise use auto-detected
        current_text = st.session_state.get("text_column", columns[default_text_idx])
        current_id = st.session_state.get("id_column", default_id)

        # Column selection
        text_col = st.selectbox(
            "Text column",
            columns,
            index=columns.index(current_text) if current_text in columns else default_text_idx,
            help="Column containing the text to analyze"
        )
        st.session_state["text_column"] = text_col

        id_col = st.selectbox(
            "ID column",
            ["(row index)"] + columns,
            index=(["(row index)"] + columns).index(current_id) if current_id in ["(row index)"] + columns else 0,
            help="Column with unique document IDs"
        )
        st.session_state["id_column"] = id_col

    st.divider()
    st.caption("Built with Streamlit")

# Main content - Introduction
st.title("Multi-LLM Topic Discovery")

# Show intro when no data loaded yet
if "data" not in st.session_state:
    st.markdown("""
    **Discover topics in your text data using multiple AI models.**

    This tool uses an ensemble of LLMs (GPT, Claude, Gemini, etc.) to identify themes
    in your documents. Using multiple models produces more robust results than any single model.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1:** Get an API key from [OpenRouter](https://openrouter.ai) (free credits for new accounts)")
    with col2:
        st.info("**Step 2:** Upload a CSV file with a text column")
    with col3:
        st.info("**Step 3:** Run Discovery → Consolidation → Assignment")

    with st.expander("📖 How does this work?", expanded=False):
        st.markdown("""
        ### Why multiple LLMs?

        Traditional topic modeling (LDA, BERTopic) struggles with domain-specific text where
        vocabulary is uniform. LLM-based discovery uses semantic understanding instead of
        word co-occurrence patterns.

        Different LLMs have different biases:
        - **Claude** tends toward psychological/mental health framing
        - **GPT** operates at higher abstraction levels
        - **Gemini** produces fine-grained distinctions
        - **DeepSeek** emphasizes power dynamics

        By combining multiple models, we get complementary perspectives that consolidate
        into a more robust taxonomy.

        ### The pipeline

        1. **Discovery**: Multiple LLMs independently identify topics from a sample of your documents
        2. **Consolidation**: A strong LLM merges semantically equivalent topics into a coherent taxonomy
        3. **Assignment**: A fast LLM assigns 1-3 topics to each document
        4. **Results**: Download your labeled data as CSV

        ### Cost

        Typical cost for a few hundred documents: **$0.50–$2.00** using cheap models.
        The app shows cost estimates before you run each step.

        ---
        *[View on GitHub](https://github.com/tomvannuenen/multi-llm-topics) • Created by Tom van Nuenen*
        """)

    st.divider()

# Pipeline overview
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; padding: 10px 0 20px 0; gap: 8px; font-size: 14px; color: #888;">
    <span style="background: #262730; padding: 6px 12px; border-radius: 6px;"><b>1.</b> Discovery</span>
    <span>→</span>
    <span style="background: #262730; padding: 6px 12px; border-radius: 6px;"><b>2.</b> Consolidation</span>
    <span>→</span>
    <span style="background: #262730; padding: 6px 12px; border-radius: 6px;"><b>3.</b> Assignment</span>
    <span>→</span>
    <span style="background: #262730; padding: 6px 12px; border-radius: 6px;"><b>4.</b> Results</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["① Discovery", "② Consolidation", "③ Assignment", "④ Results"])

# Recommended models by task
RECOMMENDED_DISCOVERY = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview-05-20",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-nano",
    "mistralai/mistral-small-3.1-24b-instruct",
]

RECOMMENDED_CONSOLIDATION = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
    "google/gemini-2.5-pro-preview-05-06",
    "anthropic/claude-3.5-sonnet",
]

RECOMMENDED_ASSIGNMENT = [
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4.1-nano",
    "anthropic/claude-haiku-4.5",
]

# Tab 1: Discovery
with tab1:
    st.header("Topic Discovery")

    with st.expander("ℹ️ How Discovery Works", expanded=False):
        st.markdown("""
        **Topic Discovery** identifies themes in your documents using multiple LLMs.

        **How sampling works:**
        - Each model processes a random sample of your documents (not the full dataset)
        - For each document, the model either assigns an existing topic or proposes a new one
        - Topics accumulate as more documents are processed—early documents create new topics, later documents mostly reuse existing ones
        - This is called "iterative discovery": the topic list grows until it stabilizes

        **Early stopping:**
        - Documents are processed in batches of 50
        - If 3 consecutive batches produce no new topics, discovery stops early
        - This saves time and money when the topic space is saturated
        - You'll see "Early stop" in the status if this happens

        **Why sample instead of processing everything?**
        - Topic discovery reaches diminishing returns quickly—after ~200-500 docs, most new documents fit existing topics
        - Sampling is much cheaper and faster than processing your entire dataset
        - The full dataset gets labeled in the Assignment step (after consolidation)

        **Why multiple models?**
        - Different LLMs have different biases and vocabularies
        - Claude tends toward psychological framing, GPT toward abstraction, Gemini toward fine-grained distinctions
        - Using 3-5 models produces more comprehensive topic coverage than any single model
        - Topics that multiple models independently discover are more robust

        **Recommended approach:**
        - Select 3-5 diverse models (different providers)
        - Use 200-500 documents per model for good coverage
        - Cheap/fast models work well for discovery (Gemini Flash, GPT-4.1-nano, Claude Haiku)

        **Cost:** Discovery is cheap (~$0.01-0.05 per 100 docs with fast models)
        """)

    with st.expander("✏️ Customize Discovery Prompt", expanded=False):
        st.caption("Edit the prompt used to discover topics. Use `{topics}` for existing topics and `{post}` for the document text.")
        edited_discovery = st.text_area(
            "Discovery Prompt",
            value=st.session_state["discovery_prompt"],
            height=300,
            key="discovery_prompt_editor",
            label_visibility="collapsed"
        )
        col_reset, col_save = st.columns([1, 1])
        with col_reset:
            if st.button("Reset to Default", key="reset_discovery"):
                st.session_state["discovery_prompt"] = DEFAULT_DISCOVERY_PROMPT
                st.rerun()
        with col_save:
            if st.button("Save Changes", key="save_discovery", type="primary"):
                st.session_state["discovery_prompt"] = edited_discovery
                st.success("Prompt saved!")

    # Fetch available models
    models_dict = get_models_by_category()
    all_models = models_dict["all"]

    # Find recommended models that exist in the available list
    discovery_defaults = [m for m in RECOMMENDED_DISCOVERY if m in all_models][:3]
    if not discovery_defaults:
        discovery_defaults = DEFAULT_DISCOVERY_MODELS[:3]

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_models = st.multiselect(
            "Select Models",
            all_models if all_models else DEFAULT_DISCOVERY_MODELS,
            default=discovery_defaults,
            help="Choose 2-5 models from different providers for diverse topic coverage. Fast/cheap models work well."
        )

        # Cap max samples at actual data size
        if "data" in st.session_state:
            max_docs = len(st.session_state["data"])
            # For small datasets, just use the full dataset
            if max_docs <= 100:
                n_samples = st.number_input(
                    "Documents per model",
                    min_value=10,
                    max_value=max_docs,
                    value=max_docs,
                    step=10,
                    help="Number of documents each model will analyze."
                )
            else:
                n_samples = st.slider(
                    "Documents per model",
                    min_value=50,
                    max_value=max_docs,
                    value=min(200, max_docs),
                    step=50,
                    help="Number of documents each model will analyze. More = better coverage but higher cost. 200-500 is usually sufficient."
                )
        else:
            n_samples = st.slider(
                "Documents per model",
                50, 1000, 200, 50,
                help="Number of documents each model will analyze. More = better coverage but higher cost. 200-500 is usually sufficient."
            )

    with col2:
        st.metric("Models Selected", len(selected_models))
        st.metric("Est. API Calls", len(selected_models) * n_samples)

        # Cost estimate
        total_cost = sum(get_model_cost_estimate(m, n_samples, "discovery") for m in selected_models)
        st.metric("Est. Cost", format_cost(total_cost))

    if st.button("🚀 Start Discovery", type="primary", use_container_width=True):
        client = get_client()
        if not client:
            st.error("Please set your OpenRouter API key")
        elif "data" not in st.session_state:
            st.error("Please upload data first")
        elif not selected_models:
            st.error("Please select at least one model")
        else:
            df = st.session_state["data"]
            text_col = st.session_state["text_column"]
            texts = df[text_col].dropna().tolist()

            all_topics = {}

            for model in selected_models:
                st.subheader(f"Running: {model.split('/')[-1]}")
                progress = st.progress(0)
                status = st.empty()

                topics = run_discovery(client, model, texts, n_samples, progress, status)

                # Merge topics
                for t, data in topics.items():
                    if t not in all_topics:
                        all_topics[t] = {"models": [], "count": 0, "description": data.get("description", "")}
                    all_topics[t]["models"].append(model.split("/")[-1])
                    all_topics[t]["count"] += data.get("count", 1)

                st.success(f"Found {len(topics)} topics")

            st.session_state["discovered_topics"] = all_topics

            st.divider()
            st.subheader("Discovery Complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Unique Topics", len(all_topics))
            col2.metric("Models Used", len(selected_models))

            # Show topics
            st.dataframe(
                pd.DataFrame([
                    {"Topic": t, "Count": d["count"], "Models": ", ".join(d["models"])}
                    for t, d in sorted(all_topics.items(), key=lambda x: -x[1]["count"])
                ]),
                use_container_width=True,
                height=400
            )

            st.info("**Next step →** Go to **② Consolidation** to merge these topics into a coherent taxonomy.")

# Tab 2: Consolidation
with tab2:
    st.header("Topic Consolidation")

    with st.expander("ℹ️ How Consolidation Works", expanded=False):
        st.markdown("""
        **Consolidation** merges the raw topics from Discovery into a coherent taxonomy.

        **Why consolidate?** Different models create different labels for the same concept
        (e.g., "communication_problems" vs "talking_issues"). A strong LLM identifies
        these duplicates and creates a unified taxonomy.

        **Recommended approach:**
        - Use a strong reasoning model (Claude Sonnet, GPT-4.1, Gemini Pro)
        - Target 50-80 final categories for most datasets
        - Review the output and adjust if needed

        **Cost:** Consolidation is a single API call (~$0.50-2.00 for 500+ topics)
        """)

    with st.expander("✏️ Customize Consolidation Prompt", expanded=False):
        st.caption("Edit the prompt used to consolidate topics. Use `{n_topics}` for topic count and `{topics}` for the topic list.")
        edited_consolidation = st.text_area(
            "Consolidation Prompt",
            value=st.session_state["consolidation_prompt"],
            height=300,
            key="consolidation_prompt_editor",
            label_visibility="collapsed"
        )
        col_reset, col_save = st.columns([1, 1])
        with col_reset:
            if st.button("Reset to Default", key="reset_consolidation"):
                st.session_state["consolidation_prompt"] = DEFAULT_CONSOLIDATION_PROMPT
                st.rerun()
        with col_save:
            if st.button("Save Changes", key="save_consolidation", type="primary"):
                st.session_state["consolidation_prompt"] = edited_consolidation
                st.success("Prompt saved!")

    if "discovered_topics" not in st.session_state:
        st.info("Run discovery first, or upload existing topics.")

        uploaded_topics = st.file_uploader(
            "Upload discovered topics (JSON)",
            type=["json"],
            key="topics_upload",
            help="JSON file from a previous discovery run"
        )
        if uploaded_topics:
            topics_data = json.load(uploaded_topics)
            if isinstance(topics_data, dict) and "topics" in topics_data:
                st.session_state["discovered_topics"] = topics_data["topics"]
            else:
                st.session_state["discovered_topics"] = topics_data
            st.success(f"Loaded {len(st.session_state['discovered_topics'])} topics")
            st.rerun()
    else:
        topics = st.session_state["discovered_topics"]

        models_dict = get_models_by_category()
        consolidation_models = models_dict["all"] if models_dict["all"] else DEFAULT_CONSOLIDATION_MODELS

        # Find best default from recommended list
        consolidation_default = next(
            (m for m in RECOMMENDED_CONSOLIDATION if m in consolidation_models),
            consolidation_models[0] if consolidation_models else None
        )
        default_idx = consolidation_models.index(consolidation_default) if consolidation_default in consolidation_models else 0

        col1, col2 = st.columns([2, 1])
        with col1:
            model = st.selectbox(
                "Consolidation Model",
                consolidation_models,
                index=default_idx,
                help="Use a strong reasoning model. Claude Sonnet or GPT-4.1 work well for semantic merging."
            )
        with col2:
            st.metric("Topics to Consolidate", len(topics))
            cost = get_model_cost_estimate(model, len(topics), "consolidation")
            st.metric("Est. Cost", format_cost(cost))

        if st.button("🔄 Consolidate Topics", type="primary", use_container_width=True):
            client = get_client()
            if not client:
                st.error("Please set your OpenRouter API key")
            else:
                progress = st.progress(0)
                status = st.empty()

                result = run_consolidation(client, model, list(topics.keys()), progress, status)

                taxonomy = result.get("taxonomy", [])
                st.session_state["taxonomy"] = taxonomy

                st.success(f"Created {len(taxonomy)} categories!")

                # Display taxonomy
                for i, cat in enumerate(taxonomy[:20], 1):
                    with st.expander(f"{i}. {cat['topic']} ({len(cat.get('source_topics', []))} sources)"):
                        st.write(cat.get("description", ""))
                        st.caption(f"Source topics: {', '.join(cat.get('source_topics', [])[:10])}...")

                st.info("**Next step →** Go to **③ Assignment** to label your documents with these topics.")

# Tab 3: Assignment
with tab3:
    st.header("Topic Assignment")

    with st.expander("ℹ️ How Assignment Works", expanded=False):
        st.markdown("""
        **Assignment** labels each document with 1-3 topics from your taxonomy.

        **How it works:** Each document is sent to the LLM along with your taxonomy.
        The model selects a primary topic and optionally 1-2 secondary topics.

        **Recommended approach:**
        - Use a fast, cheap model (Gemini Flash Lite, GPT-4.1-nano)
        - Assignment is the most expensive step (one API call per document)
        - Use 10 workers for parallel processing (built-in)

        **Cost:** ~$0.01-0.05 per 100 documents with cheap models
        """)

    with st.expander("✏️ Customize Assignment Prompt", expanded=False):
        st.caption("Edit the prompt used to assign topics. Use `{n_topics}` for topic count, `{taxonomy}` for the taxonomy, and `{post}` for the document.")
        edited_assignment = st.text_area(
            "Assignment Prompt",
            value=st.session_state["assignment_prompt"],
            height=300,
            key="assignment_prompt_editor",
            label_visibility="collapsed"
        )
        col_reset, col_save = st.columns([1, 1])
        with col_reset:
            if st.button("Reset to Default", key="reset_assignment"):
                st.session_state["assignment_prompt"] = DEFAULT_ASSIGNMENT_PROMPT
                st.rerun()
        with col_save:
            if st.button("Save Changes", key="save_assignment", type="primary"):
                st.session_state["assignment_prompt"] = edited_assignment
                st.success("Prompt saved!")

    if "taxonomy" not in st.session_state:
        st.info("Run consolidation first, or upload existing taxonomy.")

        uploaded_tax = st.file_uploader(
            "Upload taxonomy (JSON)",
            type=["json"],
            key="tax_upload",
            help="JSON file from a previous consolidation run"
        )
        if uploaded_tax:
            tax_data = json.load(uploaded_tax)
            st.session_state["taxonomy"] = tax_data.get("taxonomy", tax_data)
            st.success(f"Loaded {len(st.session_state['taxonomy'])} categories")
            st.rerun()
    else:
        taxonomy = st.session_state["taxonomy"]

        models_dict = get_models_by_category()
        assignment_models = models_dict["all"] if models_dict["all"] else DEFAULT_ASSIGNMENT_MODELS

        # Find best default from recommended list
        assignment_default = next(
            (m for m in RECOMMENDED_ASSIGNMENT if m in assignment_models),
            assignment_models[0] if assignment_models else None
        )
        default_idx = assignment_models.index(assignment_default) if assignment_default in assignment_models else 0

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            model = st.selectbox(
                "Assignment Model",
                assignment_models,
                index=default_idx,
                help="Fast, cheap models work well. This is the most API-intensive step."
            )
        with col2:
            n_docs = st.number_input(
                "Documents to process",
                10, 10000, 100, 10,
                help="Number of documents to assign topics to. Start small to verify quality."
            )
        with col3:
            st.metric("Categories", len(taxonomy))
            cost = get_model_cost_estimate(model, n_docs, "assignment")
            st.metric("Est. Cost", format_cost(cost))

        if st.button("🏷️ Assign Topics", type="primary", use_container_width=True):
            client = get_client()
            if not client:
                st.error("Please set your OpenRouter API key")
            elif "data" not in st.session_state:
                st.error("Please upload data first")
            else:
                df = st.session_state["data"]
                text_col = st.session_state["text_column"]
                id_col = st.session_state["id_column"]

                texts = df[text_col].dropna().tolist()[:n_docs]
                if id_col == "(row index)":
                    ids = list(range(len(texts)))
                else:
                    ids = df[id_col].tolist()[:n_docs]

                progress = st.progress(0)
                status = st.empty()

                results = run_assignment(client, model, texts, ids, taxonomy, progress, status)
                st.session_state["assignments"] = results

                # Convert to DataFrame
                results_df = pd.DataFrame([
                    {
                        "id": r["id"],
                        "primary_topic": r.get("primary_topic", ""),
                        "secondary_1": r.get("secondary_topics", [""])[0] if r.get("secondary_topics") else "",
                        "secondary_2": r.get("secondary_topics", ["", ""])[1] if len(r.get("secondary_topics", [])) > 1 else "",
                        "reasoning": r.get("reasoning", ""),
                        "error": r.get("error", "")
                    }
                    for r in results
                ])

                st.session_state["results_df"] = results_df
                st.success(f"Assigned topics to {len(results)} documents!")

                # Show distribution
                st.subheader("Topic Distribution")
                topic_counts = results_df["primary_topic"].value_counts().head(15)
                st.bar_chart(topic_counts)

                st.info("**Done!** Go to **④ Results** to download your data.")

# Tab 4: Results
with tab4:
    st.header("Results & Export")

    col1, col2, col3 = st.columns(3)

    # Discovery results
    with col1:
        st.subheader("📊 Discovery")
        if "discovered_topics" in st.session_state:
            topics = st.session_state["discovered_topics"]
            st.metric("Topics", len(topics))

            topics_json = json.dumps({"topics": topics}, indent=2)
            st.download_button(
                "Download Topics JSON",
                topics_json,
                "discovered_topics.json",
                "application/json"
            )
        else:
            st.caption("No discovery results yet")

    # Taxonomy
    with col2:
        st.subheader("🔄 Taxonomy")
        if "taxonomy" in st.session_state:
            taxonomy = st.session_state["taxonomy"]
            st.metric("Categories", len(taxonomy))

            tax_json = json.dumps({"taxonomy": taxonomy}, indent=2)
            st.download_button(
                "Download Taxonomy JSON",
                tax_json,
                "taxonomy.json",
                "application/json"
            )
        else:
            st.caption("No taxonomy yet")

    # Assignments
    with col3:
        st.subheader("🏷️ Assignments")
        if "results_df" in st.session_state:
            results_df = st.session_state["results_df"]
            st.metric("Documents", len(results_df))

            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "topic_assignments.csv",
                "text/csv"
            )
        else:
            st.caption("No assignments yet")

    st.divider()

    # Show data preview
    if "results_df" in st.session_state:
        st.subheader("Assignment Results Preview")
        st.dataframe(st.session_state["results_df"], use_container_width=True, height=400)


# Footer
st.divider()
st.caption("Multi-LLM Topics • Created by [Tom van Nuenen](https://tomvannuenen.github.io) • [GitHub](https://github.com/tomvannuenen/multi-llm-topics)")

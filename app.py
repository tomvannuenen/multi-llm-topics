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
# Free models have :free suffix - great for testing!
DEFAULT_DISCOVERY_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "anthropic/claude-haiku-4.5",
]

DEFAULT_CONSOLIDATION_MODELS = [
    "google/gemini-2.5-pro-exp-03-25:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "anthropic/claude-sonnet-4",
]

DEFAULT_ASSIGNMENT_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-001",
]


def fetch_ollama_models(ollama_url: str = "http://localhost:11434") -> dict:
    """Fetch available models from local Ollama instance."""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_info = {}
            for m in models:
                name = m.get("name", "")
                if name:
                    # Prefix with ollama/ to distinguish from OpenRouter models
                    model_id = f"ollama/{name}"
                    model_info[model_id] = {
                        "prompt_cost": 0,
                        "completion_cost": 0,
                        "context_length": 0,  # Ollama doesn't report this in /api/tags
                        "is_free": True,  # Local models are always free
                        "is_ollama": True,
                    }
            return model_info
    except requests.exceptions.ConnectionError:
        # Ollama not running - this is expected if user doesn't have it
        pass
    except Exception as e:
        # Only warn if Ollama is enabled
        if st.session_state.get("ollama_enabled"):
            st.warning(f"Could not connect to Ollama: {e}")
    return {}


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
                        "is_free": ":free" in model_id,
                        "is_ollama": False,
                    }
            return model_info
    except Exception as e:
        st.warning(f"Could not fetch models from OpenRouter: {e}")
    return {}


def sort_models_for_task(models: dict, task: str = "discovery") -> list:
    """Sort models: local/free first, then by price (cheap first for fast tasks, expensive first for reasoning)."""
    model_list = list(models.keys())

    def sort_key(model_id):
        info = models.get(model_id, {})
        is_ollama = model_id.startswith("ollama/") or info.get("is_ollama", False)
        is_free = is_ollama or info.get("is_free", ":free" in model_id)
        total_cost = info.get("prompt_cost", 0) + info.get("completion_cost", 0)

        if task == "consolidation":
            # For consolidation, prefer stronger (more expensive) models, but local/free first
            # Put Ollama at very top (sort key -1), then free cloud (0), then paid (1)
            priority = -1 if is_ollama else (0 if is_free else 1)
            return (priority, -total_cost)
        else:
            # For discovery/assignment, prefer cheaper models, local/free first
            priority = -1 if is_ollama else (0 if is_free else 1)
            return (priority, total_cost)

    return sorted(model_list, key=sort_key)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token)."""
    if not text or pd.isna(text):
        return 0
    return len(str(text)) // 4


def get_avg_doc_tokens(df: pd.DataFrame, text_column: str, sample_size: int = 100) -> int:
    """Get average token count from a sample of documents."""
    if text_column not in df.columns:
        return 500  # fallback
    sample = df[text_column].dropna().head(sample_size)
    if len(sample) == 0:
        return 500
    avg_chars = sample.apply(lambda x: len(str(x))).mean()
    return int(avg_chars // 4)


def get_model_cost_estimate(model_id: str, n_docs: int, task: str = "discovery",
                            avg_doc_tokens: int = None) -> float:
    """Estimate cost for a task based on model pricing."""
    models = fetch_openrouter_models()
    if model_id not in models:
        return 0.0

    pricing = models[model_id]
    prompt_cost = pricing["prompt_cost"]  # per 1M tokens
    completion_cost = pricing["completion_cost"]

    # Use actual avg tokens if provided, otherwise use defaults
    doc_tokens = avg_doc_tokens if avg_doc_tokens else 500

    if task == "discovery":
        # prompt template (~200 tokens) + doc text + ~50 tokens response
        tokens_in = n_docs * (200 + min(doc_tokens, 1500))  # cap at 1500 (we truncate long docs)
        tokens_out = n_docs * 50
    elif task == "consolidation":
        # One big call: ~20 tokens per topic input + ~50 tokens per topic output
        tokens_in = n_docs * 20  # n_docs here is n_topics
        tokens_out = n_docs * 50
    else:  # assignment
        # taxonomy (~1000 tokens) + doc text + ~100 tokens response
        tokens_in = n_docs * (1000 + min(doc_tokens, 2000))  # cap at 2000
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
    """Get models organized for different use cases, including Ollama if enabled."""
    all_models_info = fetch_openrouter_models()

    # Add Ollama models if enabled
    if st.session_state.get("ollama_enabled"):
        ollama_url = st.session_state.get("ollama_url", "http://localhost:11434")
        ollama_models = fetch_ollama_models(ollama_url)
        all_models_info.update(ollama_models)

    if not all_models_info:
        return {
            "all": DEFAULT_DISCOVERY_MODELS + DEFAULT_CONSOLIDATION_MODELS + DEFAULT_ASSIGNMENT_MODELS,
            "discovery": DEFAULT_DISCOVERY_MODELS,
            "consolidation": DEFAULT_CONSOLIDATION_MODELS,
            "assignment": DEFAULT_ASSIGNMENT_MODELS,
            "pricing": {},
        }

    # Sort models: free/local first, then by price
    all_models_sorted = sort_models_for_task(all_models_info, "discovery")

    # Categorize models (include ollama models in fast category)
    fast_models = [m for m in all_models_sorted if any(x in m.lower() for x in ["flash", "haiku", "mini", "nano", "lite", "8b", "7b", ":free", "ollama/"])]
    strong_models = [m for m in all_models_sorted if any(x in m.lower() for x in ["sonnet", "opus", "gpt-4", "claude-3", "gemini-pro", "gemini-2", ":free", "ollama/"])]

    return {
        "all": all_models_sorted,
        "discovery": fast_models if fast_models else all_models_sorted[:20],
        "consolidation": strong_models if strong_models else all_models_sorted[:10],
        "assignment": fast_models if fast_models else all_models_sorted[:20],
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


def get_client(model: str = None):
    """Get OpenAI client configured for the appropriate backend.

    Args:
        model: Model ID. If starts with 'ollama/', routes to local Ollama.
               Otherwise routes to OpenRouter.
    """
    if model and model.startswith("ollama/"):
        # Route to Ollama
        ollama_url = st.session_state.get("ollama_url", "http://localhost:11434")
        return OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")  # Ollama doesn't need real key
    else:
        # Route to OpenRouter
        api_key = st.session_state.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def get_model_for_api(model: str) -> str:
    """Strip ollama/ prefix for API calls."""
    if model.startswith("ollama/"):
        return model[7:]  # Remove "ollama/" prefix
    return model


def process_single_post(client, model, post_text, topics, lock, errors, discovery_prompt):
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

        prompt = discovery_prompt.format(topics=topics_formatted, post=post_text[:6000])

        # Get the actual model name for API (strip ollama/ prefix)
        api_model = get_model_for_api(model)

        # Try with JSON mode first, fall back to regular mode
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            # Some models don't support JSON mode, try without it
            if "json" in str(e).lower() or "response_format" in str(e).lower():
                response = client.chat.completions.create(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                )
            else:
                raise

        content = response.choices[0].message.content
        if not content:
            with lock:
                errors.append("Empty response from model")
            return "", False

        # Try to extract JSON from response (handle markdown code blocks)
        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = clean_content[clean_content.find("\n")+1:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3].strip()

        parsed = json.loads(clean_content)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}

        topic = parsed.get("topic", "")
        if isinstance(topic, str):
            topic = re.sub(r'[^a-z0-9_]', '_', topic.lower().strip())
            topic = re.sub(r'_+', '_', topic).strip('_')

        if not topic:
            with lock:
                errors.append(f"No topic in response: {content[:100]}")
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

    except json.JSONDecodeError as e:
        with lock:
            errors.append(f"JSON parse error: {str(e)[:50]}")
        return "", False
    except Exception as e:
        with lock:
            errors.append(f"Error: {str(e)[:100]}")
        return "", False


def run_discovery(client, model, texts, n_samples, progress_bar, status_text,
                  discovery_prompt, batch_size=50, early_stop_batches=3):
    """Run topic discovery on texts with early stopping."""
    sample = texts[:n_samples] if len(texts) > n_samples else texts
    topics = {}
    errors = []
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
            futures = {executor.submit(process_single_post, client, model, text, topics, lock, errors, discovery_prompt): i
                       for i, text in enumerate(batch_texts)}

            for future in as_completed(futures):
                topic, is_new = future.result()
                completed += 1
                if is_new:
                    new_topics_total += 1
                    new_in_batch += 1

                progress_bar.progress(completed / total)
                error_info = f" | {len(errors)} errors" if errors else ""
                status_text.text(f"Processed {completed}/{total} documents | {len(topics)} topics ({new_topics_total} new){error_info}")

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

    # Return errors for display
    return topics, errors


def run_consolidation(client, model, topics, progress_bar, status_text):
    """Consolidate topics into taxonomy."""
    topics_formatted = "\n".join(f"- {t}" for t in sorted(topics))
    prompt = st.session_state["consolidation_prompt"].format(n_topics=len(topics), topics=topics_formatted)

    # Get the actual model name for API (strip ollama/ prefix)
    api_model = get_model_for_api(model)

    status_text.text(f"Sending {len(topics)} topics to {format_model_name(model)}...")
    progress_bar.progress(0.3)

    response = client.chat.completions.create(
        model=api_model,
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


def run_assignment(client, model, texts, ids, taxonomy, progress_bar, status_text, assignment_prompt):
    """Assign topics to documents."""
    taxonomy_formatted = "\n".join(f"- {t['topic']}: {t.get('description', '')}" for t in taxonomy)
    valid_labels = {t['topic'] for t in taxonomy}
    n_topics = len(taxonomy)

    # Get the actual model name for API (strip ollama/ prefix)
    api_model = get_model_for_api(model)

    results = []
    total = len(texts)

    def assign_single(text, doc_id):
        prompt = assignment_prompt.format(
            n_topics=n_topics,
            taxonomy=taxonomy_formatted,
            post=text[:8000]
        )

        try:
            # Try with JSON mode first, fall back if not supported
            try:
                response = client.chat.completions.create(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                if "json" in str(e).lower() or "response_format" in str(e).lower():
                    response = client.chat.completions.create(
                        model=api_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.0,
                    )
                else:
                    raise

            content = response.choices[0].message.content
            if not content:
                return {"id": doc_id, "error": "empty_response"}

            # Handle markdown code blocks
            clean_content = content.strip()
            if clean_content.startswith("```"):
                clean_content = clean_content[clean_content.find("\n")+1:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3].strip()

            result = json.loads(clean_content)
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
            return {"id": doc_id, "error": str(e)[:100]}

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
    st.markdown("## 🏷️ Multi-LLM Topics")
    st.caption("Discover themes in your text data using generative AI models from [OpenRouter](https://openrouter.ai). Different models find different patterns—together they produce more robust results than any single model.")

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
                    # First validate the key
                    response = requests.get(
                        "https://openrouter.ai/api/v1/auth/key",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.session_state["api_key"] = api_key
                        st.session_state["api_key_validated"] = api_key
                        st.session_state["api_key_valid"] = True

                        # Try to get actual credit balance
                        credits_response = requests.get(
                            "https://openrouter.ai/api/v1/credits",
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10
                        )
                        if credits_response.status_code == 200:
                            credits_data = credits_response.json().get("data", {})
                            total_credits = credits_data.get("total_credits", 0)
                            total_usage = credits_data.get("total_usage", 0)
                            remaining = total_credits - total_usage
                            st.success(f"✓ Valid — ${remaining:.2f} credits remaining")
                        else:
                            # Fall back to key info
                            data = response.json().get("data", {})
                            usage = data.get("usage", 0)
                            limit = data.get("limit")
                            if limit is not None:
                                remaining = limit - usage
                                st.success(f"✓ Valid — ${remaining:.2f} credits remaining")
                            else:
                                st.success("✓ API key valid")
                    else:
                        st.session_state["api_key_valid"] = False
                        st.error("Invalid API key")
                except Exception as e:
                    st.session_state["api_key_valid"] = False
                    st.error(f"Could not validate key: {e}")
        elif st.session_state.get("api_key_valid"):
            st.success("✓ API key valid")

    st.divider()

    # Ollama (local models) configuration
    with st.expander("🖥️ Local Models (Ollama)", expanded=False):
        st.caption("Run models locally for free using [Ollama](https://ollama.com)")

        # Quick check if Ollama is reachable (to detect if running locally vs cloud)
        ollama_url = st.session_state.get("ollama_url", "http://localhost:11434")
        ollama_available = False
        try:
            test_response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            ollama_available = test_response.status_code == 200
        except:
            pass

        if not ollama_available:
            # Show message for online/cloud users
            st.info("**Ollama requires running the app locally.**")
            st.markdown("""
            To use free local models:
            1. Clone the repo: `git clone https://github.com/tomvannuenen/multi-llm-topics`
            2. Install [Ollama](https://ollama.com) and pull models
            3. Run: `streamlit run app.py`

            See the [README](https://github.com/tomvannuenen/multi-llm-topics#running-locally-with-ollama-free) for detailed instructions.
            """)
        else:
            # Ollama is available - show enable checkbox
            ollama_enabled = st.checkbox(
                "Enable Ollama",
                value=st.session_state.get("ollama_enabled", False),
                help="Connect to a local Ollama instance for free local models"
            )
            st.session_state["ollama_enabled"] = ollama_enabled

            if ollama_enabled:
                ollama_url = st.text_input(
                    "Ollama URL",
                    value=ollama_url,
                    help="URL of your Ollama instance"
                )
                st.session_state["ollama_url"] = ollama_url

                # Show available models
                ollama_models = fetch_ollama_models(ollama_url)
                if ollama_models:
                    st.success(f"✓ Connected — {len(ollama_models)} models available")
                    model_names = [m.replace("ollama/", "") for m in ollama_models.keys()]
                    st.caption(f"Models: {', '.join(model_names[:5])}{'...' if len(model_names) > 5 else ''}")

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

    # Show completed steps
    steps_done = []
    if "data" in st.session_state:
        steps_done.append("✓ Data loaded")
    if "discovered_topics" in st.session_state:
        steps_done.append("✓ Discovery done")
    if "taxonomy" in st.session_state:
        steps_done.append("✓ Consolidated")
    if "assignments" in st.session_state:
        steps_done.append("✓ Assigned")

    if steps_done:
        st.caption(" → ".join(steps_done))

# Main content

# Show intro when no data loaded yet
if "data" not in st.session_state:
    st.markdown("""
    **Discover topics in your text data using multiple AI models.**

    This tool uses an ensemble of LLMs (GPT, Claude, Gemini, etc.) to identify themes
    in your documents. Using multiple models produces more robust results than any single model.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔑 Step 1")
        st.caption("Get an API key from [OpenRouter](https://openrouter.ai) — free to sign up")
    with col2:
        st.markdown("#### 📄 Step 2")
        st.caption("Upload a CSV file with a text column, or load the sample data")
    with col3:
        st.markdown("#### ▶️ Step 3")
        st.caption("Run Discovery → Consolidation → Assignment, then download results")

    st.success("💡 **Try for free:** Models with `:free` in the name cost nothing. Or enable [Ollama](https://ollama.com) in the sidebar to run models locally for free.")

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

tab1, tab2, tab3, tab4 = st.tabs(["① Discovery", "② Consolidation", "③ Assignment", "④ Results"])

# Recommended models by task (free models first for easy testing)
RECOMMENDED_DISCOVERY = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-nano",
]

RECOMMENDED_CONSOLIDATION = [
    "google/gemini-2.5-pro-exp-03-25:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
]

RECOMMENDED_ASSIGNMENT = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
]


def format_model_name(model_id: str) -> str:
    """Format model ID for display - shorter, more readable."""
    if model_id.startswith("ollama/"):
        # Show Ollama models with [local] indicator
        name = model_id[7:]  # Remove "ollama/" prefix
        return f"{name} [local]"
    # Remove provider prefix for display, keep :free suffix visible
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    return name

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
        - Fast/cheap models work well — look for models with `:free` suffix

        **Cost:** Discovery is cheap (~$0.01-0.05 per 100 docs), or free with `:free` models
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
            format_func=format_model_name,
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

        # Cost estimate based on actual text length
        avg_tokens = None
        if "data" in st.session_state and "text_column" in st.session_state:
            avg_tokens = get_avg_doc_tokens(st.session_state["data"], st.session_state["text_column"])
        total_cost = sum(get_model_cost_estimate(m, n_samples, "discovery", avg_tokens) for m in selected_models)
        st.metric("Est. Cost", format_cost(total_cost))

    if st.button("🚀 Start Discovery", type="primary", use_container_width=True):
        # Check if any OpenRouter models are selected (need API key)
        openrouter_models = [m for m in selected_models if not m.startswith("ollama/")]
        if openrouter_models and not get_client():
            st.error("Please set your OpenRouter API key for cloud models")
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
                st.subheader(f"Running: {format_model_name(model)}")
                progress = st.progress(0)
                status = st.empty()

                # Get client for this specific model (routes to Ollama or OpenRouter)
                client = get_client(model)
                topics, errors = run_discovery(client, model, texts, n_samples, progress, status,
                                               st.session_state["discovery_prompt"])

                # Show errors if any
                if errors:
                    with st.expander(f"⚠️ {len(errors)} errors occurred", expanded=False):
                        for err in errors[:10]:  # Show first 10
                            st.caption(err)
                        if len(errors) > 10:
                            st.caption(f"... and {len(errors) - 10} more")

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
        - Use a strong reasoning model (larger models work better for semantic merging)
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
                format_func=format_model_name,
                help="Use a strong reasoning model. Larger models work better for semantic merging."
            )
        with col2:
            st.metric("Topics to Consolidate", len(topics))
            cost = get_model_cost_estimate(model, len(topics), "consolidation")
            st.metric("Est. Cost", format_cost(cost))

        if st.button("🔄 Consolidate Topics", type="primary", use_container_width=True):
            client = get_client(model)
            if not client:
                st.error("Please set your OpenRouter API key for cloud models")
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
        - Use a fast, cheap model — or a `:free` model for testing
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
                format_func=format_model_name,
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
            # Cost estimate based on actual text length
            avg_tokens = None
            if "data" in st.session_state and "text_column" in st.session_state:
                avg_tokens = get_avg_doc_tokens(st.session_state["data"], st.session_state["text_column"])
            cost = get_model_cost_estimate(model, n_docs, "assignment", avg_tokens)
            st.metric("Est. Cost", format_cost(cost))

        if st.button("🏷️ Assign Topics", type="primary", use_container_width=True):
            client = get_client(model)
            if not client:
                st.error("Please set your OpenRouter API key for cloud models")
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

                results = run_assignment(client, model, texts, ids, taxonomy, progress, status,
                                         st.session_state["assignment_prompt"])
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

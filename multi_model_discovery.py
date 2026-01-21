#!/usr/bin/env python3
"""
Multi-Model Topic Discovery.

Runs topic discovery with multiple LLMs to produce robust,
cross-validated topic taxonomies.

Usage:
    # List available models
    python multi_model_discovery.py --list-models

    # Run with recommended diverse models
    python multi_model_discovery.py --input posts.parquet --text-column body --recommended

    # Run with specific models
    python multi_model_discovery.py --input posts.parquet --text-column body \
        --models google/gemini-2.0-flash-001 anthropic/claude-haiku-4.5

    # Customize iterations and samples
    python multi_model_discovery.py --input posts.parquet --text-column body \
        --models MODEL1 MODEL2 --iterations 3 --samples 500
"""

import argparse
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests
import pandas as pd
from openai import OpenAI

# Provider region mapping for cross-cultural analysis
PROVIDER_REGIONS = {
    "openai": "US",
    "anthropic": "US",
    "google": "US",
    "meta-llama": "US",
    "mistralai": "Europe",
    "deepseek": "China",
    "qwen": "China",
    "alibaba": "China",
    "cohere": "Canada",
    "ai21": "Israel",
}

# Recommended models for diverse provider coverage
RECOMMENDED_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-nano",
    "google/gemini-2.0-flash-001",
    "mistralai/ministral-8b-2412",
    "deepseek/deepseek-chat-v3-0324",
]

# Prompt for topic discovery (structured JSON output)
DISCOVERY_PROMPT = """You will receive a post and a set of existing topic categories. Your task is to identify the core topic of this post.

[Existing Topics]
{topics}

[Instructions]
1. Read the post and identify its PRIMARY topic - the main issue or situation being discussed.
2. Topic labels must be GENERALIZABLE (not specific to this post).
3. If the post fits an existing topic well, use that topic.
4. Only create a NEW topic if no existing topic adequately captures the core issue.
5. Use lowercase_with_underscores format for topic labels (e.g., "communication_breakdown").

[Post]
{post}

Respond with JSON only:
{{"action": "existing" or "new", "topic": "topic_label", "description": "Brief description (only if new topic)"}}"""


def get_available_models() -> list[dict]:
    """Fetch available models from OpenRouter API."""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"}
        )
        if response.status_code == 200:
            return response.json().get("data", [])
    except Exception as e:
        print(f"Warning: Could not fetch models: {e}")
    return []


def filter_frontier_models(models: list[dict]) -> list[dict]:
    """Filter to frontier/recommended models suitable for topic discovery."""
    frontier = []
    for m in models:
        model_id = m.get("id", "")
        # Filter criteria
        if any(x in model_id.lower() for x in ["preview", "beta", "deprecated", "free"]):
            continue
        if m.get("context_length", 0) < 8000:
            continue
        # Prefer chat/instruct models
        if any(x in model_id.lower() for x in ["chat", "instruct", "turbo", "flash", "haiku", "sonnet"]):
            frontier.append(m)
    return frontier


def list_models():
    """List available models grouped by provider."""
    models = get_available_models()
    frontier = filter_frontier_models(models)

    by_provider = defaultdict(list)
    for m in frontier:
        provider = m["id"].split("/")[0]
        by_provider[provider].append(m)

    print("\nAvailable Models by Provider:")
    print("=" * 60)
    for provider, provider_models in sorted(by_provider.items()):
        region = PROVIDER_REGIONS.get(provider, "Unknown")
        print(f"\n{provider.upper()} ({region}):")
        for m in sorted(provider_models, key=lambda x: x["id"])[:5]:
            price_in = m.get("pricing", {}).get("prompt", 0)
            price_out = m.get("pricing", {}).get("completion", 0)
            try:
                price_str = f"${float(price_in)*1e6:.2f}/${float(price_out)*1e6:.2f} per M"
            except (ValueError, TypeError):
                price_str = "pricing unavailable"
            print(f"  {m['id']:<45} {price_str}")

    print("\n" + "=" * 60)
    print("Recommended models for multi-model discovery:")
    for m in RECOMMENDED_MODELS:
        print(f"  {m}")


def process_single_post(client: OpenAI, model: str, post_text: str,
                        topics: dict, lock: threading.Lock) -> tuple[str, bool]:
    """Process a single post and return (topic, is_new)."""
    topics_formatted = "\n".join(f"- {t}: {d.get('description', '')}"
                                  for t, d in topics.items()) if topics else "(No topics yet)"

    prompt = DISCOVERY_PROMPT.format(
        topics=topics_formatted,
        post=post_text[:6000]
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "https://github.com/multi-llm-topics",
                "X-Title": "Multi-LLM Topic Discovery"
            }
        )

        content = response.choices[0].message.content
        if not content:
            return "", False

        result = content.strip()
        parsed = json.loads(result)

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if not isinstance(parsed, dict):
            return "", False

        topic = parsed.get("topic", "")
        if isinstance(topic, str):
            topic = topic.strip()
        else:
            topic = str(topic) if topic else ""

        if not topic:
            return "", False

        # Normalize topic label
        topic = re.sub(r'[^a-z0-9_]', '_', topic.lower())
        topic = re.sub(r'_+', '_', topic).strip('_')

        action = parsed.get("action", "existing")
        description = parsed.get("description", "")

        with lock:
            if action == "new" and topic not in topics:
                topics[topic] = {
                    "description": description,
                    "count": 1
                }
                return topic, True
            elif topic in topics:
                topics[topic]["count"] = topics[topic].get("count", 0) + 1
                return topic, False
            else:
                # New topic even if action was "existing" (topic didn't exist)
                topics[topic] = {
                    "description": description,
                    "count": 1
                }
                return topic, True

    except Exception as e:
        return "", False


def run_discovery(client: OpenAI, model: str, posts_df: pd.DataFrame,
                  text_column: str, output_dir: Path, iteration: int,
                  n_samples: int = 500, n_workers: int = 10,
                  early_stop_batches: int = 2, batch_size: int = 50) -> dict:
    """Run topic discovery for a single model and iteration."""

    sample = posts_df.sample(n=min(n_samples, len(posts_df)),
                             random_state=42 + iteration)

    topics = {}
    lock = threading.Lock()
    posts_processed = 0
    batches_without_new = 0
    early_stopped = False

    total_posts = len(sample)
    print(f"      Processing {total_posts} posts in batches of {batch_size}...")

    for batch_start in range(0, total_posts, batch_size):
        batch_end = min(batch_start + batch_size, total_posts)
        batch = sample.iloc[batch_start:batch_end]

        new_topics_in_batch = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_single_post, client, model,
                               row[text_column], topics, lock)
                for _, row in batch.iterrows()
            ]

            for future in as_completed(futures):
                topic, is_new = future.result()
                posts_processed += 1
                if is_new:
                    new_topics_in_batch += 1

        # Check early stopping
        if new_topics_in_batch == 0:
            batches_without_new += 1
            if batches_without_new >= early_stop_batches:
                early_stopped = True
                print(f"      Early stop: {batches_without_new} batches without new topics")
                break
        else:
            batches_without_new = 0

        print(f"      Batch {batch_start//batch_size + 1}: {new_topics_in_batch} new topics, {len(topics)} total")

    # Save results
    safe_model = model.replace("/", "_").replace(":", "_")
    output_file = output_dir / f"{safe_model}_iteration_{iteration}.json"

    result = {
        "model_id": model,
        "iteration": iteration,
        "posts_processed": posts_processed,
        "early_stopped": early_stopped,
        "topics": topics,
        "n_topics": len(topics),
        "timestamp": datetime.now().isoformat()
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Topic Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_model_discovery.py --list-models
  python multi_model_discovery.py --input data.parquet --text-column text --recommended
  python multi_model_discovery.py --input data.csv --text-column content --models MODEL1 MODEL2
        """
    )
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--input", type=str,
                        help="Input file (parquet or csv)")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column containing text to analyze")
    parser.add_argument("--output", type=str, default="./topic_discovery",
                        help="Output directory for results")
    parser.add_argument("--recommended", action="store_true",
                        help="Use recommended diverse models")
    parser.add_argument("--models", nargs="+",
                        help="Model IDs to use")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per model")
    parser.add_argument("--samples", type=int, default=500,
                        help="Posts to sample per iteration")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for early stopping")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.input:
        parser.error("--input is required (unless using --list-models)")

    # Load data
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found. Available: {list(df.columns)}")

    print(f"Loaded {len(df)} documents from {input_path}")

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get models
    if args.recommended:
        models = RECOMMENDED_MODELS
    elif args.models:
        models = args.models
    else:
        parser.error("Specify --models or --recommended")

    print(f"Models: {models}")
    print(f"Iterations: {args.iterations}")
    print(f"Samples per iteration: {args.samples}")

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # Run discovery
    all_results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for iteration in range(1, args.iterations + 1):
            print(f"\n  Iteration {iteration}/{args.iterations}")
            result = run_discovery(
                client, model, df, args.text_column, output_dir,
                iteration, args.samples, args.workers, batch_size=args.batch_size
            )
            all_results.append(result)
            print(f"    → {result['n_topics']} topics discovered")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_topics = set()
    for r in all_results:
        all_topics.update(r["topics"].keys())

    print(f"\nTotal unique topics across all models: {len(all_topics)}")
    print("\nPer model:")
    by_model = defaultdict(set)
    for r in all_results:
        model = r["model_id"].split("/")[-1]
        by_model[model].update(r["topics"].keys())

    for model, topics in sorted(by_model.items(), key=lambda x: -len(x[1])):
        print(f"  {model}: {len(topics)} topics")

    # Save combined results
    combined_file = output_dir / f"discovery_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, "w") as f:
        json.dump({
            "models": models,
            "iterations": args.iterations,
            "samples_per_iteration": args.samples,
            "total_unique_topics": len(all_topics),
            "results": all_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

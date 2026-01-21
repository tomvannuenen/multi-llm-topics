#!/usr/bin/env python3
"""
Multi-Label Topic Assignment.

Assigns 1-3 topics from a taxonomy to each document.
Returns primary topic (most relevant) plus optional secondary topics.

Usage:
    python assign_topics.py --input posts.parquet --taxonomy taxonomy.json --text-column body
    python assign_topics.py --input data.csv --taxonomy topics.json --sample 100
    python assign_topics.py --input posts.parquet --taxonomy topics.json --resume
    python assign_topics.py --input posts.parquet --taxonomy topics.json --model anthropic/claude-haiku-4.5
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

# Default model (Gemini Flash Lite is cheap and works well for classification)
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

ASSIGNMENT_PROMPT = """You are assigning topic labels to a document.

Given the document below, select 1-3 topics from the taxonomy that best describe its content.

Rules:
- Select the PRIMARY topic (most relevant) first
- Add 1-2 SECONDARY topics only if clearly applicable (not just tangentially related)
- Focus on the core subject matter, not minor details
- If only one topic applies, that's fine

TAXONOMY ({n_topics} topics):
{taxonomy}

DOCUMENT:
{post}

Respond with JSON only:
{{
  "primary_topic": "most_relevant_topic_label",
  "secondary_topics": ["other_topic_1", "other_topic_2"],
  "reasoning": "Brief explanation of why these topics apply"
}}"""


def load_taxonomy(taxonomy_file: Path) -> tuple[list[dict], set[str], str]:
    """Load taxonomy and format for prompt."""
    with open(taxonomy_file) as f:
        data = json.load(f)

    taxonomy = data.get("taxonomy", data.get("details", []))
    valid_labels = set()
    lines = []

    for cat in taxonomy:
        label = cat.get("topic", cat.get("label", ""))
        desc = cat.get("description", "")
        valid_labels.add(label)
        lines.append(f"- {label}: {desc}")

    return taxonomy, valid_labels, "\n".join(lines)


def load_posts(input_file: Path, text_column: str, sample: int = None) -> pd.DataFrame:
    """Load posts from parquet or CSV, optionally sampling."""
    if input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    elif input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    return df


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load existing checkpoint."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed": {}}


def save_checkpoint(checkpoint: dict, checkpoint_file: Path):
    """Save checkpoint."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)


def assign_topics_to_post(client: OpenAI, model: str, post_id: str,
                          post_text: str, taxonomy_formatted: str,
                          valid_labels: set, n_topics: int) -> dict:
    """Assign topics to a single post."""

    # Truncate very long posts
    if len(post_text) > 8000:
        post_text = post_text[:8000] + "..."

    prompt = ASSIGNMENT_PROMPT.format(
        n_topics=n_topics,
        taxonomy=taxonomy_formatted,
        post=post_text
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "https://github.com/multi-llm-topics",
                "X-Title": "Topic Assignment"
            }
        )

        content = response.choices[0].message.content
        if not content:
            return {"post_id": post_id, "error": "empty_response"}

        result = json.loads(content.strip())

        primary = result.get("primary_topic", "")
        secondary = result.get("secondary_topics", [])
        reasoning = result.get("reasoning", "")

        # Validate primary topic
        if primary not in valid_labels:
            # Try fuzzy match
            primary_norm = primary.lower().replace(" ", "_").replace("-", "_")
            matches = [l for l in valid_labels
                       if primary_norm in l.lower() or l.lower() in primary_norm]
            primary = matches[0] if matches else "unknown"

        # Validate secondary topics
        valid_secondary = [t for t in secondary if t in valid_labels]

        return {
            "post_id": post_id,
            "primary_topic": primary,
            "secondary_topics": valid_secondary,
            "all_topics": [primary] + valid_secondary,
            "n_topics": 1 + len(valid_secondary),
            "reasoning": reasoning
        }

    except json.JSONDecodeError as e:
        return {"post_id": post_id, "error": f"json_parse: {e}"}
    except Exception as e:
        return {"post_id": post_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Multi-label topic assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python assign_topics.py --input posts.parquet --taxonomy taxonomy.json --text-column body
  python assign_topics.py --input data.csv --taxonomy topics.json --sample 100
  python assign_topics.py --input posts.parquet --taxonomy topics.json --resume
  python assign_topics.py --input posts.parquet --taxonomy topics.json --model anthropic/claude-haiku-4.5
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input file (parquet or csv)")
    parser.add_argument("--taxonomy", type=str, required=True,
                        help="Taxonomy JSON file from consolidation")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column containing text to analyze")
    parser.add_argument("--id-column", type=str, default="id",
                        help="Column containing document IDs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to input directory)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model for assignment")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N posts (for testing)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N posts")
    args = parser.parse_args()

    # Setup paths
    input_file = Path(args.input)
    taxonomy_file = Path(args.taxonomy)
    output_dir = Path(args.output) if args.output else input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "assignment_checkpoint.json"

    print("=" * 60)
    print("MULTI-LABEL TOPIC ASSIGNMENT")
    print("=" * 60)

    # Load taxonomy
    print("\n1. Loading taxonomy...")
    taxonomy, valid_labels, taxonomy_formatted = load_taxonomy(taxonomy_file)
    print(f"   {len(valid_labels)} topics loaded from {taxonomy_file}")

    # Load posts
    print("\n2. Loading posts...")
    df = load_posts(input_file, args.text_column, args.sample)
    print(f"   {len(df)} posts loaded from {input_file}")

    # Resume handling
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else {"completed": {}}
    already_done = set(checkpoint.get("completed", {}).keys())
    if already_done:
        print(f"   Resuming: {len(already_done)} already completed")

    remaining = df[~df[args.id_column].isin(already_done)]
    print(f"   {len(remaining)} posts to process")

    if len(remaining) == 0:
        print("\n   All posts already processed!")
        return

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # Process posts
    print(f"\n3. Assigning topics with {args.model}...")
    results = list(checkpoint.get("completed", {}).values())
    errors = 0
    processed = 0

    id_col = args.id_column
    text_col = args.text_column

    def process_post(row):
        return assign_topics_to_post(
            client, args.model,
            row[id_col], row[text_col],
            taxonomy_formatted, valid_labels, len(valid_labels)
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_post, row): row[id_col]
            for _, row in remaining.iterrows()
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            checkpoint["completed"][result["post_id"]] = result
            processed += 1

            if "error" in result:
                errors += 1

            if processed % 50 == 0:
                print(f"   Processed {processed}/{len(remaining)} ({errors} errors)")

            if processed % args.checkpoint_every == 0:
                save_checkpoint(checkpoint, checkpoint_file)

    save_checkpoint(checkpoint, checkpoint_file)
    print(f"\n   Complete: {len(results)} posts, {errors} errors")

    # Build output DataFrame
    print("\n4. Building output...")

    results_rows = []
    for r in results:
        if "error" in r:
            results_rows.append({
                "post_id": r["post_id"],
                "primary_topic": "",
                "secondary_topic_1": "",
                "secondary_topic_2": "",
                "n_topics": 0,
                "all_topics_json": "[]",
                "reasoning": "",
                "error": r.get("error", "")
            })
        else:
            sec = r.get("secondary_topics", [])
            results_rows.append({
                "post_id": r["post_id"],
                "primary_topic": r.get("primary_topic", ""),
                "secondary_topic_1": sec[0] if len(sec) > 0 else "",
                "secondary_topic_2": sec[1] if len(sec) > 1 else "",
                "n_topics": r.get("n_topics", 1),
                "all_topics_json": json.dumps(r.get("all_topics", [])),
                "reasoning": r.get("reasoning", ""),
                "error": ""
            })

    results_df = pd.DataFrame(results_rows)

    # Merge with posts
    output_df = df.merge(results_df, left_on=id_col, right_on='post_id', how='left')

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model = args.model.replace("/", "_").replace(":", "_")

    parquet_file = output_dir / f"posts_with_topics_{safe_model}_{timestamp}.parquet"
    output_df.to_parquet(parquet_file)
    print(f"   Saved: {parquet_file}")

    # CSV with key columns
    csv_cols = [id_col, 'primary_topic', 'secondary_topic_1', 'secondary_topic_2', 'n_topics']
    csv_cols = [c for c in csv_cols if c in output_df.columns]
    csv_file = output_dir / f"posts_with_topics_{safe_model}_{timestamp}.csv"
    output_df[csv_cols].to_csv(csv_file, index=False)
    print(f"   CSV: {csv_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid = results_df[results_df['error'] == '']
    print(f"\nTopic count distribution:")
    for n in [1, 2, 3]:
        count = (valid['n_topics'] == n).sum()
        pct = count / len(valid) * 100 if len(valid) > 0 else 0
        print(f"   {n} topic(s): {count} ({pct:.1f}%)")

    print(f"\nTop 15 primary topics:")
    for topic, count in valid['primary_topic'].value_counts().head(15).items():
        pct = count / len(valid) * 100
        print(f"   {topic}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

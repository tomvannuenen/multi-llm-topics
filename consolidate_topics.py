#!/usr/bin/env python3
"""
Topic Consolidation.

Consolidates raw topics from multi-model discovery into a coherent taxonomy
using a strong LLM.

Usage:
    python consolidate_topics.py --input ./topic_discovery
    python consolidate_topics.py --input ./topic_discovery --output ./consolidated
    python consolidate_topics.py --input ./topic_discovery --model anthropic/claude-sonnet-4
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

CONSOLIDATION_PROMPT = """You are consolidating topic labels from multiple LLMs into a coherent taxonomy.

These topics were discovered by different LLMs analyzing the same corpus. Due to different naming conventions and granularity preferences, there is significant overlap. Your task is to:

1. Merge topics that represent the SAME underlying concept (different wording for same idea)
2. Keep topics SEPARATE if they represent meaningfully different contexts
3. Aim for 50-80 final topics that cover the full range of issues

IMPORTANT DISTINCTIONS TO PRESERVE:
- Severity differences (e.g., "boundary violations" vs "abuse/control")
- Cause differences (e.g., "jealousy from insecurity" vs "jealousy from evidence")
- Context differences (e.g., "breakup considerations" vs "post-breakup recovery")

CRITICAL: You MUST map EVERY SINGLE ONE of the {n_topics} source topics to a category.
Do not give examples - provide the COMPLETE list of source topics for each category.
Every topic from the input list must appear exactly once in a source_topics array.

For each final topic, provide:
- A clear, consistent label (lowercase_with_underscores, 2-5 words)
- A brief description (1 sentence)
- ALL source topics it consolidates (complete list, not examples)

RAW TOPICS TO CONSOLIDATE ({n_topics} total):
{topics}

Respond with JSON:
{{
  "taxonomy": [
    {{
      "topic": "topic_label",
      "description": "What this topic covers",
      "source_topics": ["ALL", "MATCHING", "TOPICS", "FROM", "INPUT", "..."]
    }},
    ...
  ],
  "unmapped": ["any topics that genuinely don't fit - should be very few or none"]
}}"""


def load_all_topics(input_dir: Path) -> tuple[dict, dict]:
    """Load all topics from discovery files."""
    files = list(input_dir.glob("*_iteration_*.json"))
    if not files:
        files = [f for f in input_dir.glob("*.json")
                 if "combined" not in f.name and "consolidated" not in f.name]

    all_topics = {}
    model_counts = {}

    for f in files:
        with open(f) as fp:
            d = json.load(fp)

        model_id = d.get("model_id", "unknown")
        short = model_id.split("/")[-1]

        if short not in model_counts:
            model_counts[short] = set()

        for topic in d.get("topics", {}).keys():
            if topic not in all_topics:
                all_topics[topic] = {"models": set(), "count": 0}
            all_topics[topic]["models"].add(short)
            all_topics[topic]["count"] += 1
            model_counts[short].add(topic)

    return all_topics, model_counts


def consolidate_topics(client: OpenAI, model: str, topics: list[str]) -> dict:
    """Run LLM consolidation on topics."""
    topics_formatted = "\n".join(f"- {t}" for t in sorted(topics))

    prompt = CONSOLIDATION_PROMPT.format(
        n_topics=len(topics),
        topics=topics_formatted
    )

    print(f"Sending {len(topics)} topics to {model}...")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_headers={
            "HTTP-Referer": "https://github.com/multi-llm-topics",
            "X-Title": "Topic Consolidation"
        }
    )

    result = response.choices[0].message.content

    print(f"Response length: {len(result) if result else 0} chars")
    if not result:
        print("WARNING: Empty response from model")
        return {"taxonomy": [], "unmapped": []}

    # Strip markdown code blocks if present
    clean_result = result.strip()
    if clean_result.startswith("```"):
        first_newline = clean_result.find("\n")
        clean_result = clean_result[first_newline + 1:]
        if clean_result.endswith("```"):
            clean_result = clean_result[:-3].strip()

    try:
        parsed = json.loads(clean_result)
        taxonomy = parsed.get('taxonomy', [])
        print(f"Parsed successfully: {len(taxonomy)} categories")

        if not taxonomy:
            print(f"WARNING: No taxonomy found. Keys in response: {list(parsed.keys())}")

        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response (first 500 chars): {clean_result[:500]}")
        return {"taxonomy": [], "unmapped": []}


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate multi-model topics into taxonomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consolidate_topics.py --input ./topic_discovery
  python consolidate_topics.py --input ./discovery --output ./consolidated
  python consolidate_topics.py --input ./discovery --model openai/gpt-4.5-turbo
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing discovery results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to input dir)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4",
                        help="Model to use for consolidation")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TOPIC CONSOLIDATION")
    print("=" * 60)

    # Load topics
    print("\n1. Loading discovered topics...")
    all_topics, model_counts = load_all_topics(input_dir)

    if not all_topics:
        print(f"ERROR: No topics found in {input_dir}")
        return

    print(f"   Total unique topics: {len(all_topics)}")
    print(f"   From models:")
    for model, topics in sorted(model_counts.items(), key=lambda x: -len(x[1])):
        print(f"     - {model}: {len(topics)} topics")

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # Run consolidation
    print(f"\n2. Consolidating with {args.model}...")
    topic_list = list(all_topics.keys())
    result = consolidate_topics(client, args.model, topic_list)

    # Analyze results
    taxonomy = result.get("taxonomy", [])
    unmapped = result.get("unmapped", [])

    print(f"\n3. Results:")
    print(f"   Final topics: {len(taxonomy)}")
    print(f"   Unmapped: {len(unmapped)}")

    # Calculate coverage
    mapped_topics = set()
    for t in taxonomy:
        mapped_topics.update(t.get("source_topics", []))
    coverage = len(mapped_topics) / len(all_topics) * 100 if all_topics else 0
    print(f"   Coverage: {len(mapped_topics)}/{len(all_topics)} ({coverage:.1f}%)")

    # Show taxonomy
    print(f"\n" + "=" * 60)
    print("CONSOLIDATED TAXONOMY")
    print("=" * 60)

    for i, t in enumerate(taxonomy, 1):
        n_sources = len(t.get("source_topics", []))
        print(f"\n{i:2d}. {t['topic']}")
        print(f"    {t.get('description', '')}")
        print(f"    Sources: {n_sources} topics")

    # Save results
    print(f"\n4. Saving results...")

    safe_model = args.model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "input_topics": len(all_topics),
        "output_topics": len(taxonomy),
        "coverage_pct": coverage,
        "taxonomy": taxonomy,
        "unmapped": unmapped,
        "source_models": list(model_counts.keys())
    }

    output_file = output_dir / f"consolidated_{safe_model}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   Saved to: {output_file}")

    # Also save just the taxonomy for easy reference
    taxonomy_file = output_dir / f"taxonomy_{len(taxonomy)}.json"
    with open(taxonomy_file, "w") as f:
        json.dump({
            "n_topics": len(taxonomy),
            "topics": [t["topic"] for t in taxonomy],
            "taxonomy": taxonomy
        }, f, indent=2, ensure_ascii=False)

    print(f"   Taxonomy: {taxonomy_file}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

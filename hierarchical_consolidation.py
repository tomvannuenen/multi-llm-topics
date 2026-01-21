#!/usr/bin/env python3
"""
Hierarchical Topic Consolidation.

Uses a two-stage approach to consolidate large numbers of topics:
1. Split topics into manageable batches (~150-200 each)
2. LLM consolidates each batch independently → intermediate categories
3. Final LLM pass merges intermediate categories → final taxonomy

This avoids the "too many topics" problem while giving LLM full authority
over semantic distinctions (unlike embedding-driven approaches).

Usage:
    python hierarchical_consolidation.py --input ./topic_discovery
    python hierarchical_consolidation.py --input ./discovery --output ./consolidated
    python hierarchical_consolidation.py --input ./discovery --batches 5 --model anthropic/claude-sonnet-4
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

BATCH_CONSOLIDATION_PROMPT = """You are consolidating topic labels discovered from a text corpus.

Your task is to group these topics into coherent categories based on semantic similarity and thematic overlap. Topics should be in the same category only if they represent the same underlying concept.

IMPORTANT PRINCIPLES:
- Merge topics that are just different wordings for the same concept
- Preserve distinctions where the topics represent meaningfully different ideas
- Consider both surface similarity and deeper semantic meaning
- Aim for categories that would be useful for classification

For each category:
- Provide a clear label (lowercase_with_underscores, 2-5 words)
- Brief description (1 sentence)
- List ALL topics that belong to it

TOPICS TO CONSOLIDATE ({n_topics} total):
{topics}

Respond with JSON only:
{{
  "categories": [
    {{
      "label": "category_label",
      "description": "What this category covers",
      "topics": ["topic1", "topic2", ...]
    }}
  ]
}}"""

FINAL_CONSOLIDATION_PROMPT = """You are performing the final consolidation of topic categories.

These categories were created by consolidating raw topics in batches. Your task is to:
1. MERGE categories that represent the same concept (different wording for same idea)
2. KEEP categories SEPARATE if they represent meaningfully different concepts
3. Aim for 60-80 final categories that cover the full range of topics

IMPORTANT: Preserve meaningful distinctions:
- Severity or intensity differences
- Causal or contextual differences
- Different implications or use cases

INTERMEDIATE CATEGORIES ({n_categories} total):
{categories}

For each final category, provide:
- A clear label (lowercase_with_underscores, 2-5 words)
- Brief description
- List of ALL source categories it consolidates
- List of ALL original topics (combine from source categories)

Respond with JSON only:
{{
  "taxonomy": [
    {{
      "label": "final_category_label",
      "description": "What this category covers",
      "source_categories": ["intermediate_cat1", "intermediate_cat2"],
      "topics": ["all", "original", "topics", "from", "sources"]
    }}
  ]
}}"""


def load_all_topics(input_dir: Path) -> list[str]:
    """Load all unique topics from discovery files."""
    files = list(input_dir.glob("*_iteration_*.json"))
    if not files:
        files = [f for f in input_dir.glob("*.json")
                 if "combined" not in f.name and "consolidated" not in f.name]

    all_topics = set()
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        for topic in d.get("topics", {}).keys():
            all_topics.add(topic)

    return sorted(list(all_topics))


def split_into_batches(topics: list[str], n_batches: int, seed: int = 42) -> list[list[str]]:
    """Split topics into n roughly equal batches."""
    random.seed(seed)
    shuffled = topics.copy()
    random.shuffle(shuffled)

    batch_size = len(shuffled) // n_batches
    batches = []

    for i in range(n_batches):
        start = i * batch_size
        if i == n_batches - 1:
            # Last batch gets any remainder
            batches.append(shuffled[start:])
        else:
            batches.append(shuffled[start:start + batch_size])

    return batches


def consolidate_batch(client: OpenAI, model: str, topics: list[str],
                      batch_num: int) -> dict:
    """Consolidate a single batch of topics."""
    topics_formatted = "\n".join(f"- {t}" for t in sorted(topics))
    prompt = BATCH_CONSOLIDATION_PROMPT.format(
        n_topics=len(topics),
        topics=topics_formatted
    )

    print(f"   Batch {batch_num}: sending {len(topics)} topics...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "https://github.com/multi-llm-topics",
                "X-Title": "Batch Consolidation"
            }
        )

        content = response.choices[0].message.content
        if not content:
            print(f"   Batch {batch_num}: empty response")
            return {"categories": [], "batch": batch_num}

        # Strip markdown if present
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean[clean.find("\n") + 1:]
            if clean.endswith("```"):
                clean = clean[:-3].strip()

        result = json.loads(clean)
        categories = result.get("categories", [])
        print(f"   Batch {batch_num}: {len(categories)} categories")

        # Add batch number for tracking
        for cat in categories:
            cat["batch"] = batch_num

        return {"categories": categories, "batch": batch_num}

    except Exception as e:
        print(f"   Batch {batch_num}: error - {e}")
        return {"categories": [], "batch": batch_num, "error": str(e)}


def final_consolidation(client: OpenAI, model: str,
                        intermediate_categories: list[dict]) -> dict:
    """Perform final consolidation of intermediate categories."""

    # Format categories for prompt
    cat_lines = []
    for cat in intermediate_categories:
        cat_lines.append(f"- {cat['label']}: {cat.get('description', '')} ({len(cat.get('topics', []))} topics)")

    categories_formatted = "\n".join(cat_lines)

    prompt = FINAL_CONSOLIDATION_PROMPT.format(
        n_categories=len(intermediate_categories),
        categories=categories_formatted
    )

    print(f"\n4. Final consolidation of {len(intermediate_categories)} intermediate categories...")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_headers={
            "HTTP-Referer": "https://github.com/multi-llm-topics",
            "X-Title": "Final Consolidation"
        }
    )

    content = response.choices[0].message.content
    if not content:
        return {"taxonomy": []}

    # Strip markdown if present
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean[clean.find("\n") + 1:]
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    return json.loads(clean)


def map_topics_to_final(intermediate_categories: list[dict],
                        final_taxonomy: list[dict]) -> list[dict]:
    """Map original topics to final categories using intermediate mapping."""

    # Build intermediate label -> topics mapping
    intermediate_map = {}
    for cat in intermediate_categories:
        intermediate_map[cat["label"]] = cat.get("topics", [])

    # For each final category, collect all topics from its source categories
    for final_cat in final_taxonomy:
        all_topics = set()
        source_cats = final_cat.get("source_categories", [])

        for source_label in source_cats:
            # Try exact match first
            if source_label in intermediate_map:
                all_topics.update(intermediate_map[source_label])
            else:
                # Try fuzzy match (in case of minor label differences)
                for int_label, int_topics in intermediate_map.items():
                    if source_label.lower() in int_label.lower() or int_label.lower() in source_label.lower():
                        all_topics.update(int_topics)

        final_cat["topics"] = sorted(list(all_topics))

    return final_taxonomy


def verify_coverage(original_topics: list[str], final_taxonomy: list[dict]) -> dict:
    """Verify all original topics are mapped."""
    mapped = set()
    for cat in final_taxonomy:
        mapped.update(cat.get("topics", []))

    original_set = set(original_topics)
    covered = mapped & original_set
    missing = original_set - mapped
    extra = mapped - original_set

    return {
        "original_count": len(original_set),
        "mapped_count": len(covered),
        "coverage_pct": len(covered) / len(original_set) * 100 if original_set else 0,
        "missing": sorted(list(missing)),
        "extra": sorted(list(extra))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical topic consolidation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hierarchical_consolidation.py --input ./topic_discovery
  python hierarchical_consolidation.py --input ./discovery --output ./consolidated
  python hierarchical_consolidation.py --input ./discovery --batches 5 --model anthropic/claude-sonnet-4
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing discovery results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to input dir)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4",
                        help="Model for consolidation")
    parser.add_argument("--batches", type=int, default=5,
                        help="Number of batches for stage 1")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel workers for batch processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for batch splitting")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HIERARCHICAL TOPIC CONSOLIDATION")
    print("=" * 60)

    # Load topics
    print("\n1. Loading discovered topics...")
    topics = load_all_topics(input_dir)
    print(f"   Loaded {len(topics)} unique topics from {input_dir}")

    # Split into batches
    print(f"\n2. Splitting into {args.batches} batches...")
    batches = split_into_batches(topics, args.batches, args.seed)
    for i, batch in enumerate(batches, 1):
        print(f"   Batch {i}: {len(batch)} topics")

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # Stage 1: Consolidate each batch
    print(f"\n3. Stage 1: Consolidating batches with {args.model}...")
    intermediate_categories = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(consolidate_batch, client, args.model, batch, i + 1): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            result = future.result()
            intermediate_categories.extend(result.get("categories", []))

    print(f"\n   Stage 1 complete: {len(intermediate_categories)} intermediate categories")

    # Save intermediate results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    intermediate_file = output_dir / f"hierarchical_intermediate_{timestamp}.json"
    with open(intermediate_file, "w") as f:
        json.dump({
            "model": args.model,
            "n_batches": args.batches,
            "categories": intermediate_categories
        }, f, indent=2)
    print(f"   Saved intermediate: {intermediate_file}")

    # Stage 2: Final consolidation
    final_result = final_consolidation(client, args.model, intermediate_categories)
    final_taxonomy = final_result.get("taxonomy", [])
    print(f"   Stage 2 complete: {len(final_taxonomy)} final categories")

    # Map topics to final categories
    print("\n5. Mapping original topics to final categories...")
    final_taxonomy = map_topics_to_final(intermediate_categories, final_taxonomy)

    # Verify coverage
    coverage = verify_coverage(topics, final_taxonomy)
    print(f"   Coverage: {coverage['mapped_count']}/{coverage['original_count']} ({coverage['coverage_pct']:.1f}%)")
    if coverage['missing']:
        print(f"   Missing: {len(coverage['missing'])} topics")

    # Save final results
    safe_model = args.model.replace("/", "_").replace(":", "_")
    output = {
        "method": "hierarchical_consolidation",
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "stages": {
            "stage1_batches": args.batches,
            "stage1_intermediate_categories": len(intermediate_categories),
            "stage2_final_categories": len(final_taxonomy)
        },
        "input_topics": len(topics),
        "output_categories": len(final_taxonomy),
        "coverage": coverage,
        "taxonomy": final_taxonomy
    }

    output_file = output_dir / f"hierarchical_{safe_model}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n6. Saved to: {output_file}")

    # Also save clean taxonomy
    taxonomy_file = output_dir / f"taxonomy_hierarchical_{len(final_taxonomy)}.json"
    with open(taxonomy_file, "w") as f:
        json.dump({
            "method": "hierarchical_consolidation",
            "model": args.model,
            "n_categories": len(final_taxonomy),
            "taxonomy": [
                {
                    "label": cat["label"],
                    "description": cat.get("description", ""),
                    "n_topics": len(cat.get("topics", []))
                }
                for cat in final_taxonomy
            ],
            "full_taxonomy": final_taxonomy
        }, f, indent=2, ensure_ascii=False)
    print(f"   Taxonomy: {taxonomy_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL TAXONOMY SUMMARY")
    print("=" * 60)

    # Sort by topic count
    sorted_tax = sorted(final_taxonomy, key=lambda x: -len(x.get("topics", [])))
    for i, cat in enumerate(sorted_tax[:20], 1):
        n_topics = len(cat.get("topics", []))
        print(f"{i:2d}. {cat['label']} ({n_topics} topics)")
        print(f"    {cat.get('description', '')[:80]}")

    if len(final_taxonomy) > 20:
        print(f"\n... and {len(final_taxonomy) - 20} more categories")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

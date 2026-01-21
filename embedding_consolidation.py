#!/usr/bin/env python3
"""
Embedding-Driven Topic Consolidation.

Clusters discovered topics using embeddings, then has an LLM label each cluster.
This provides a comparison/alternative to pure LLM-driven consolidation.

Usage:
    python embedding_consolidation.py --input ./topic_discovery
    python embedding_consolidation.py --input ./discovery --target-clusters 70
    python embedding_consolidation.py --input ./discovery --output ./consolidated --threshold 0.65
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from openai import OpenAI

LABELING_PROMPT = """You are labeling a cluster of related topic labels that were discovered from a text corpus.

These topics were grouped together because they have similar semantic meaning. Your task is to:
1. Identify the common theme
2. Provide a single consolidated label (lowercase_with_underscores, 2-5 words)
3. Provide a brief description

Topics in this cluster:
{topics}

Respond with JSON only:
{{"label": "consolidated_label", "description": "Brief description of what this cluster covers"}}"""


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


def find_threshold_for_target(embeddings: np.ndarray, target_clusters: int,
                               tolerance: int = 5) -> tuple[float, int]:
    """Binary search for threshold that gives approximately target_clusters."""
    low, high = 0.1, 0.9
    best_threshold = 0.5
    best_n = 0

    for _ in range(20):  # Binary search iterations
        mid = (low + high) / 2
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=mid,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        n_clusters = len(set(labels))

        if abs(n_clusters - target_clusters) < abs(best_n - target_clusters):
            best_threshold = mid
            best_n = n_clusters

        if abs(n_clusters - target_clusters) <= tolerance:
            return mid, n_clusters
        elif n_clusters > target_clusters:
            low = mid  # Need more merging, increase threshold
        else:
            high = mid  # Need less merging, decrease threshold

    return best_threshold, best_n


def cluster_topics(topics: list[str], embeddings: np.ndarray,
                   threshold: float = None, target_clusters: int = None) -> dict:
    """Cluster topics using agglomerative clustering."""

    if target_clusters and not threshold:
        threshold, actual_n = find_threshold_for_target(embeddings, target_clusters)
        print(f"   Found threshold {threshold:.3f} for ~{target_clusters} clusters (actual: {actual_n})")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Group topics by cluster
    clusters = {}
    for topic, label in zip(topics, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(topic)

    return clusters, threshold


def label_cluster(client: OpenAI, model: str, topics: list[str]) -> dict:
    """Have LLM label a single cluster."""
    topics_formatted = "\n".join(f"- {t}" for t in topics)
    prompt = LABELING_PROMPT.format(topics=topics_formatted)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "https://github.com/multi-llm-topics",
                "X-Title": "Cluster Labeling"
            }
        )

        content = response.choices[0].message.content
        if not content:
            return {"label": "unknown", "description": "Failed to generate label"}

        result = json.loads(content.strip())
        return result
    except Exception as e:
        print(f"   Error labeling cluster: {e}")
        return {"label": "error", "description": str(e)}


def label_all_clusters(client: OpenAI, model: str, clusters: dict,
                       n_workers: int = 10) -> list[dict]:
    """Label all clusters in parallel."""
    taxonomy = []

    def process_cluster(cluster_id, topics):
        result = label_cluster(client, model, topics)
        return {
            "cluster_id": int(cluster_id),  # Convert numpy int64 to Python int
            "label": result.get("label", "unknown"),
            "description": result.get("description", ""),
            "source_topics": topics,
            "size": len(topics)
        }

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_cluster, cid, topics): cid
            for cid, topics in clusters.items()
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            taxonomy.append(result)
            if (i + 1) % 10 == 0:
                print(f"   Labeled {i + 1}/{len(clusters)} clusters...")

    # Sort by size (largest first)
    taxonomy.sort(key=lambda x: -x["size"])
    return taxonomy


def compare_taxonomies(embedding_taxonomy: list[dict], llm_taxonomy_path: Path) -> dict:
    """Compare embedding-driven taxonomy to LLM-driven taxonomy."""

    # Load LLM taxonomy
    with open(llm_taxonomy_path) as f:
        llm_data = json.load(f)

    llm_taxonomy = llm_data.get("taxonomy", llm_data.get("details", []))

    # Build topic -> category mappings
    embedding_map = {}
    for cat in embedding_taxonomy:
        for topic in cat["source_topics"]:
            embedding_map[topic] = cat["label"]

    llm_map = {}
    for cat in llm_taxonomy:
        for topic in cat.get("source_topics", []):
            llm_map[topic] = cat["topic"]

    # Find topics in both
    common_topics = set(embedding_map.keys()) & set(llm_map.keys())

    # Analyze agreement/disagreement
    same_category = 0
    different_category = []

    for topic in common_topics:
        emb_cat = embedding_map[topic]
        llm_cat = llm_map[topic]

        # Check if they're similar (not exact match, but related)
        if emb_cat == llm_cat:
            same_category += 1
        else:
            different_category.append({
                "topic": topic,
                "embedding_category": emb_cat,
                "llm_category": llm_cat
            })

    # Find distinctions preserved by LLM but collapsed by embedding
    # (topics in same embedding cluster but different LLM categories)
    llm_preserved = []
    for cat in embedding_taxonomy:
        topics = cat["source_topics"]
        llm_cats_in_cluster = set()
        for t in topics:
            if t in llm_map:
                llm_cats_in_cluster.add(llm_map[t])

        if len(llm_cats_in_cluster) > 1:
            llm_preserved.append({
                "embedding_cluster": cat["label"],
                "llm_categories": list(llm_cats_in_cluster),
                "sample_topics": topics[:5]
            })

    # Find distinctions preserved by embedding but collapsed by LLM
    embedding_preserved = []
    for cat in llm_taxonomy:
        topics = cat.get("source_topics", [])
        emb_cats_in_cluster = set()
        for t in topics:
            if t in embedding_map:
                emb_cats_in_cluster.add(embedding_map[t])

        if len(emb_cats_in_cluster) > 1:
            embedding_preserved.append({
                "llm_category": cat["topic"],
                "embedding_clusters": list(emb_cats_in_cluster),
                "sample_topics": topics[:5]
            })

    return {
        "common_topics": len(common_topics),
        "exact_agreement": same_category,
        "agreement_rate": same_category / len(common_topics) if common_topics else 0,
        "llm_preserves_distinctions": len(llm_preserved),
        "embedding_preserves_distinctions": len(embedding_preserved),
        "llm_preserved_examples": llm_preserved[:10],
        "embedding_preserved_examples": embedding_preserved[:10],
        "disagreements_sample": different_category[:20]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Embedding-driven topic consolidation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embedding_consolidation.py --input ./topic_discovery
  python embedding_consolidation.py --input ./discovery --target-clusters 70
  python embedding_consolidation.py --input ./discovery --output ./consolidated --threshold 0.65
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing discovery results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to input dir)")
    parser.add_argument("--compare-taxonomy", type=str, default=None,
                        help="Path to LLM taxonomy JSON for comparison")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine distance threshold for clustering")
    parser.add_argument("--target-clusters", type=int, default=70,
                        help="Target number of clusters (will find appropriate threshold)")
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-001",
                        help="Model to use for cluster labeling")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers for labeling")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EMBEDDING-DRIVEN TOPIC CONSOLIDATION")
    print("=" * 60)

    # Load topics
    print("\n1. Loading discovered topics...")
    topics = load_all_topics(input_dir)
    print(f"   Loaded {len(topics)} unique topics from {input_dir}")

    # Embed topics
    print("\n2. Computing embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(topics, show_progress_bar=True)
    print(f"   Embedding shape: {embeddings.shape}")

    # Cluster
    print("\n3. Clustering topics...")
    clusters, threshold = cluster_topics(
        topics, embeddings,
        threshold=args.threshold,
        target_clusters=args.target_clusters
    )
    print(f"   Created {len(clusters)} clusters at threshold {threshold:.3f}")

    # Show cluster size distribution
    sizes = [len(v) for v in clusters.values()]
    print(f"   Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}")

    # Label clusters with LLM
    print(f"\n4. Labeling clusters with {args.model}...")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    taxonomy = label_all_clusters(client, args.model, clusters, args.workers)
    print(f"   Labeled {len(taxonomy)} clusters")

    # Compare to LLM-driven taxonomy
    print("\n5. Comparing to LLM-driven consolidation...")
    llm_taxonomy_path = Path(args.compare_taxonomy) if args.compare_taxonomy else None

    comparison = None
    if llm_taxonomy_path and llm_taxonomy_path.exists():
        comparison = compare_taxonomies(taxonomy, llm_taxonomy_path)
        print(f"   Common topics: {comparison['common_topics']}")
        print(f"   Exact label agreement: {comparison['exact_agreement']} ({comparison['agreement_rate']:.1%})")
        print(f"   Distinctions LLM preserves that embedding collapses: {comparison['llm_preserves_distinctions']}")
        print(f"   Distinctions embedding preserves that LLM collapses: {comparison['embedding_preserves_distinctions']}")
    else:
        print("   No LLM taxonomy provided for comparison (use --compare-taxonomy)")

    # Save results
    print("\n6. Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output = {
        "method": "embedding_driven",
        "timestamp": datetime.now().isoformat(),
        "embedding_model": "all-MiniLM-L6-v2",
        "clustering": {
            "method": "agglomerative",
            "linkage": "average",
            "metric": "cosine",
            "threshold": float(threshold),  # Convert numpy float to Python float
            "target_clusters": args.target_clusters
        },
        "labeling_model": args.model,
        "input_topics": len(topics),
        "output_clusters": len(taxonomy),
        "taxonomy": taxonomy,
        "comparison_to_llm": comparison
    }

    output_file = output_dir / f"embedding_consolidation_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"   Saved to: {output_file}")

    # Print sample taxonomy
    print("\n" + "=" * 60)
    print("SAMPLE TAXONOMY (top 15 by size)")
    print("=" * 60)
    for i, cat in enumerate(taxonomy[:15], 1):
        print(f"\n{i:2d}. {cat['label']} ({cat['size']} topics)")
        print(f"    {cat['description']}")
        print(f"    Examples: {', '.join(cat['source_topics'][:3])}")

    # Print comparison insights
    if comparison:
        print("\n" + "=" * 60)
        print("KEY DIFFERENCES FROM LLM CONSOLIDATION")
        print("=" * 60)

        print("\nDistinctions LLM preserves that embedding collapses:")
        for ex in comparison["llm_preserved_examples"][:5]:
            print(f"\n  Embedding cluster: {ex['embedding_cluster']}")
            print(f"  LLM splits into: {', '.join(ex['llm_categories'][:4])}")

        print("\nDistinctions embedding preserves that LLM collapses:")
        for ex in comparison["embedding_preserved_examples"][:5]:
            print(f"\n  LLM category: {ex['llm_category']}")
            print(f"  Embedding splits into: {', '.join(ex['embedding_clusters'][:4])}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

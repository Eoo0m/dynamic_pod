import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import random


def load_data():
    """Load playlist tracks and embeddings data"""
    # Load playlist tracks
    df = pd.read_csv(
        "/Users/eomjoonseo/projects/dynamic_pod/preprocessing/playlist_tracks_filtered.csv"
    )

    # Load embeddings and keys
    embeddings = np.load(
        "/Users/eomjoonseo/projects/dynamic_pod/contrastive_top5pct_embeddings.npy"
    )
    keys = np.load(
        "/Users/eomjoonseo/projects/dynamic_pod/contrastive_top5pct_keys.npy"
    )

    print(f"Loaded {len(df)} track records")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Keys shape: {keys.shape}")

    return df, embeddings, keys


def create_track_embedding_map(embeddings, keys):
    """Create a mapping from track_id to embedding"""
    # Convert keys to string if they're bytes
    if isinstance(keys[0], bytes):
        keys = [k.decode("utf-8") for k in keys]
    elif isinstance(keys[0], np.bytes_):
        keys = [str(k, "utf-8") for k in keys]

    track_embedding_map = {}
    for i, key in enumerate(keys):
        track_embedding_map[key] = embeddings[i]

    return track_embedding_map


def sample_playlists(df, n_playlists=50, min_tracks=10):
    """Sample playlists for analysis"""
    # Get playlists with sufficient tracks
    playlist_counts = df.groupby("playlist_id").size()
    valid_playlists = playlist_counts[playlist_counts >= min_tracks].index.tolist()

    # Sample playlists
    sampled_playlists = random.sample(
        valid_playlists, min(n_playlists, len(valid_playlists))
    )

    return df[df["playlist_id"].isin(sampled_playlists)]


def analyze_position_similarity(df_sample, track_embedding_map):
    """Analyze similarity vs position distance within playlists"""
    results = []

    for playlist_id in df_sample["playlist_id"].unique():
        playlist_tracks = df_sample[
            df_sample["playlist_id"] == playlist_id
        ].sort_values("track_index")
        track_ids = playlist_tracks["track_id"].tolist()

        # Skip playlists with tracks not in embedding
        available_tracks = [tid for tid in track_ids if tid in track_embedding_map]
        if len(available_tracks) < 5:  # Need at least 5 tracks for meaningful analysis
            continue

        # Get embeddings for available tracks in order
        embeddings_list = []
        positions = []
        track_ids_filtered = []

        for i, track_id in enumerate(track_ids):
            if track_id in track_embedding_map:
                embeddings_list.append(track_embedding_map[track_id])
                positions.append(i)
                track_ids_filtered.append(track_id)

        if len(embeddings_list) < 5:
            continue

        embeddings_array = np.array(embeddings_list)

        # Calculate pairwise similarities and position distances (max distance 20)
        for i in range(len(embeddings_array)):
            for j in range(i + 1, len(embeddings_array)):
                position_distance = abs(positions[i] - positions[j])

                # Only analyze distances up to 20
                if position_distance <= 20:
                    similarity = cosine_similarity(
                        [embeddings_array[i]], [embeddings_array[j]]
                    )[0][0]

                    results.append(
                        {
                            "playlist_id": playlist_id,
                            "track1_pos": positions[i],
                            "track2_pos": positions[j],
                            "position_distance": position_distance,
                            "similarity": similarity,
                            "track1_id": track_ids_filtered[i],
                            "track2_id": track_ids_filtered[j],
                        }
                    )

    return pd.DataFrame(results)


def analyze_similarity_by_distance(results_df):
    """Analyze similarity statistics by position distance"""
    stats_by_distance = (
        results_df.groupby("position_distance")["similarity"]
        .agg(["count", "mean", "std", "var", "min", "max", "median"])
        .round(4)
    )

    return stats_by_distance


def compute_correlations(results_df):
    """Compute correlation statistics"""
    pearson_corr, pearson_p = pearsonr(
        results_df["position_distance"], results_df["similarity"]
    )
    spearman_corr, spearman_p = spearmanr(
        results_df["position_distance"], results_df["similarity"]
    )

    # Get detailed statistics by distance
    stats_by_distance = analyze_similarity_by_distance(results_df)

    # Adjacent vs distant comparison
    adjacent_sim = results_df[results_df["position_distance"] == 1]["similarity"]
    distant_sim = results_df[results_df["position_distance"] >= 15]["similarity"]

    from scipy import stats

    t_stat, t_p = stats.ttest_ind(adjacent_sim, distant_sim)

    print("=== PLAYLIST POSITION-SIMILARITY ANALYSIS ===\n")
    print(f"Total track pairs analyzed: {len(results_df):,}")
    print(f"Number of playlists: {results_df['playlist_id'].nunique()}")
    print(f"\nCorrelation Analysis:")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")

    print(f"\nAdjacent vs Distant Tracks:")
    print(
        f"Adjacent tracks (distance=1) - Mean similarity: {adjacent_sim.mean():.4f} (±{adjacent_sim.std():.4f})"
    )
    print(
        f"Distant tracks (distance>=15) - Mean similarity: {distant_sim.mean():.4f} (±{distant_sim.std():.4f})"
    )
    print(f"T-test p-value: {t_p:.4e}")

    print(f"\nSimilarity Statistics by Position Distance (1-20):")
    print("=" * 80)
    print(
        f"{'Distance':<8} {'Count':<8} {'Mean':<8} {'Std':<8} {'Var':<8} {'Min':<8} {'Max':<8} {'Median':<8}"
    )
    print("-" * 80)

    for distance in range(1, 21):
        if distance in stats_by_distance.index:
            row = stats_by_distance.loc[distance]
            print(
                f"{distance:<8} {int(row['count']):<8} {row['mean']:<8.4f} {row['std']:<8.4f} {row['var']:<8.4f} {row['min']:<8.4f} {row['max']:<8.4f} {row['median']:<8.4f}"
            )
        else:
            print(
                f"{distance:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}"
            )

    print("=" * 80)

    return {
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "adjacent_mean_similarity": adjacent_sim.mean(),
        "distant_mean_similarity": distant_sim.mean(),
        "ttest_p_value": t_p,
        "stats_by_distance": stats_by_distance,
    }


def main():
    print("Loading data...")
    df, embeddings, keys = load_data()

    print("Creating track-embedding mapping...")
    track_embedding_map = create_track_embedding_map(embeddings, keys)
    print(f"Mapped {len(track_embedding_map)} tracks to embeddings")

    print("Sampling playlists...")
    df_sample = sample_playlists(df, n_playlists=500, min_tracks=10)
    print(
        f"Sampled {df_sample['playlist_id'].nunique()} playlists with {len(df_sample)} tracks"
    )

    print("Analyzing position-similarity relationships...")
    results_df = analyze_position_similarity(df_sample, track_embedding_map)
    print(f"Generated {len(results_df)} track pairs for analysis")

    if len(results_df) > 0:
        print("Computing statistics...")
        compute_correlations(results_df)

        # Save results
        results_df.to_csv(
            "/Users/eomjoonseo/projects/dynamic_pod/statistics/playlist_position_similarity_results.csv",
            index=False,
        )
        print("\nResults saved to playlist_position_similarity_results.csv")
    else:
        print("No valid results generated. Check data alignment.")


if __name__ == "__main__":
    main()

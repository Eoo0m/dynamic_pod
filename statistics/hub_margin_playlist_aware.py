import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import defaultdict


def load_data():
    """Load all required data"""
    df = pd.read_csv(
        "/Users/eomjoonseo/projects/dynamic_pod/preprocessing/playlist_tracks_filtered.csv"
    )
    embeddings = np.load(
        "/Users/eomjoonseo/projects/dynamic_pod/contrastive_top5pct_embeddings.npy"
    )
    keys = np.load(
        "/Users/eomjoonseo/projects/dynamic_pod/contrastive_top5pct_keys.npy"
    )
    neighbors_df = pd.read_csv(
        "/Users/eomjoonseo/projects/dynamic_pod/statistics/neighbors_counts.csv"
    )

    print(f"Playlist data: {len(df)} records")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Neighbors data: {len(neighbors_df)} tracks")
    return df, embeddings, keys, neighbors_df


def create_track_embedding_map(embeddings, keys):
    """track_id → embedding 매핑"""
    if isinstance(keys[0], (bytes, np.bytes_)):
        keys = [
            k.decode("utf-8") if isinstance(k, (bytes, np.bytes_)) else str(k)
            for k in keys
        ]
    return {key: embeddings[i] for i, key in enumerate(keys)}


def create_playlist_track_map(df):
    """track_id → 포함된 playlist_id 집합 매핑"""
    track_to_playlists = defaultdict(set)
    for _, row in df.iterrows():
        track_to_playlists[row["track_id"]].add(row["playlist_id"])
    return track_to_playlists


def compute_intra_inter(
    track_ids, track_embedding_map, track_to_playlists, n_samples=300
):
    """같은 플리 / 다른 플리 평균 유사도 계산"""
    valid_tracks = [tid for tid in track_ids if tid in track_embedding_map]
    if len(valid_tracks) < 10:
        return None, None

    intra_sims, inter_sims = [], []
    attempts = 0
    while (
        len(intra_sims) < n_samples or len(inter_sims) < n_samples
    ) and attempts < n_samples * 20:
        t1, t2 = random.sample(valid_tracks, 2)
        emb1, emb2 = track_embedding_map[t1], track_embedding_map[t2]
        sim = cosine_similarity([emb1], [emb2])[0][0]
        if track_to_playlists[t1].intersection(track_to_playlists[t2]):
            if len(intra_sims) < n_samples:
                intra_sims.append(sim)
        else:
            if len(inter_sims) < n_samples:
                inter_sims.append(sim)
        attempts += 1

    if len(intra_sims) == 0 or len(inter_sims) == 0:
        return None, None
    return np.mean(intra_sims), np.mean(inter_sims)


def analyze_by_quantile(df, neighbors_df, track_embedding_map, n_bins=5):
    """neighbors_count 분위별 intra/inter/margin 평균 계산"""
    track_to_playlists = create_playlist_track_map(df)
    neighbors_df["hub_level"] = pd.qcut(
        neighbors_df["neighbors_count"],
        q=n_bins,
        labels=[f"Q{i+1}" for i in range(n_bins)],
        duplicates="drop",
    )

    results = []
    for level in neighbors_df["hub_level"].dropna().unique():
        level_tracks = neighbors_df[neighbors_df["hub_level"] == level][
            "track_id"
        ].tolist()
        intra, inter = compute_intra_inter(
            level_tracks, track_embedding_map, track_to_playlists, n_samples=300
        )
        if intra is None or inter is None:
            continue
        margin = intra - inter
        results.append(
            {
                "hub_level": level,
                "track_count": len(level_tracks),
                "intra_mean": intra,
                "inter_mean": inter,
                "margin_mean": margin,
            }
        )
    return pd.DataFrame(results)


def main():
    random.seed(42)
    df, embeddings, keys, neighbors_df = load_data()
    track_embedding_map = create_track_embedding_map(embeddings, keys)
    results_df = analyze_by_quantile(df, neighbors_df, track_embedding_map, n_bins=5)

    print("\n=== 분위별 평균 유사도 요약 ===")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "intra_mean": "{:.4f}".format,
                "inter_mean": "{:.4f}".format,
                "margin_mean": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()

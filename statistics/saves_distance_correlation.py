import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances
import argparse


def load_embeddings(path="contrastive_learning/contrastive_top5pct_win_embeddings.npy"):
    """임베딩 데이터 로드"""
    return np.load(path)


def load_meta(path="contrastive_learning/contrastive_top5pct_win_meta.csv"):
    """메타데이터 로드"""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_playlist_data(path="crawler/playlist_tracks.csv"):
    """플레이리스트 데이터 로드"""
    return pd.read_csv(path)


def compute_all_playlist_distances(embeddings, meta, playlist_data):
    """모든 플레이리스트의 내부 평균 거리 계산"""
    # track_id를 인덱스로 매핑
    track_to_idx = {track_id: idx for idx, track_id in enumerate(meta["track_id"])}

    playlist_results = []

    # 플레이리스트별로 그룹화
    for playlist_id, group in playlist_data.groupby("playlist_id"):
        track_ids = group["track_id"].tolist()

        # 임베딩에 있는 트랙들만 필터링
        valid_indices = []
        for track_id in track_ids:
            if track_id in track_to_idx:
                valid_indices.append(track_to_idx[track_id])

        if len(valid_indices) < 2:
            continue

        # 해당 트랙들의 임베딩
        playlist_embeddings = embeddings[valid_indices]

        # 모든 트랙 쌍의 코사인 거리 계산
        distances = cosine_distances(playlist_embeddings)

        # 상삼각행렬에서 평균 거리 계산 (자기 자신 제외)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        avg_distance = np.mean(upper_triangle)

        # 플레이리스트의 saves 수
        saves = 0
        if "saves" in group.columns:
            try:
                saves_val = group["saves"].iloc[0]
                if pd.notna(saves_val) and saves_val != "":
                    saves = int(float(saves_val))
            except:
                saves = 0

        playlist_results.append(
            {
                "playlist_id": playlist_id,
                "avg_distance": avg_distance,
                "saves": saves,
                "size": len(valid_indices),
            }
        )

    return pd.DataFrame(playlist_results)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between saves and playlist distances"
    )
    parser.add_argument(
        "--emb", default="contrastive_learning/contrastive_top5pct_win_embeddings.npy"
    )
    parser.add_argument(
        "--meta", default="contrastive_learning/contrastive_top5pct_win_meta.csv"
    )
    parser.add_argument("--playlist", default="crawler/playlist_tracks.csv")
    parser.add_argument("--output", default="statistics/saves_distance_correlation.png")
    args = parser.parse_args()

    print("데이터 로딩 중...")
    embeddings = load_embeddings(args.emb)
    meta = load_meta(args.meta)
    playlist_data = load_playlist_data(args.playlist)

    print(f"임베딩 shape: {embeddings.shape}")
    print(f"메타데이터 shape: {meta.shape}")
    print(f"플레이리스트 데이터 shape: {playlist_data.shape}")

    # 1. 모든 플레이리스트의 내부 평균 거리 계산
    print("\n모든 플레이리스트의 내부 평균 거리 계산 중...")
    playlist_results = compute_all_playlist_distances(embeddings, meta, playlist_data)

    # 2. 전체 플레이리스트 내부 평균 거리 통계
    all_distances = playlist_results["avg_distance"].values
    print(f"\n=== 모든 플레이리스트 내부 평균 거리 통계 ===")
    print(f"분석된 플레이리스트 수: {len(all_distances):,}개")
    print(f"평균 거리 범위: {all_distances.min():.4f} ~ {all_distances.max():.4f}")
    print(
        f"평균 거리 전체 평균: {all_distances.mean():.4f} ± {all_distances.std():.4f}"
    )
    print(f"평균 거리 중앙값: {np.median(all_distances):.4f}")

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n평균 거리 분위수:")
    for p in percentiles:
        print(f"{p:2d}th percentile: {np.percentile(all_distances, p):.4f}")

    # 3. saves가 있는 플레이리스트만 필터링
    playlist_with_saves = playlist_results[playlist_results["saves"] > 0].copy()
    print(f"\nSaves가 있는 플레이리스트: {len(playlist_with_saves):,}개")

    if len(playlist_with_saves) > 0:
        # saves 구간별 평균 내부 거리 통계
        saves_values = playlist_with_saves["saves"].values
        distance_values = playlist_with_saves["avg_distance"].values

        # saves를 구간별로 나누기
        quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
        bins = np.quantile(saves_values, quantiles)
        bins = np.unique(bins)

        print(f"\n=== Saves 구간별 평균 내부 거리 통계 ===")
        print("Saves Range\t\t\tCount\tAvg Saves\tAvg Distance\tStd Distance")
        print("-" * 80)

        for i in range(len(bins) - 1):
            mask = (saves_values >= bins[i]) & (saves_values < bins[i + 1])
            if i == len(bins) - 2:  # 마지막 구간은 최댓값 포함
                mask = (saves_values >= bins[i]) & (saves_values <= bins[i + 1])

            if mask.sum() > 0:
                bin_distances = distance_values[mask]
                bin_saves = saves_values[mask]
                print(
                    f"{int(bins[i]):,}-{int(bins[i+1]):,}\t\t{mask.sum():<6}\t{np.mean(bin_saves):<8.0f}\t{np.mean(bin_distances):<10.4f}\t{np.std(bin_distances):.4f}"
                )

        # 상관관계 계산
        pearson_corr, pearson_p = pearsonr(saves_values, distance_values)
        spearman_corr, spearman_p = spearmanr(saves_values, distance_values)

        print(f"\nSaves vs Distance 상관관계:")
        print(f"Pearson: {pearson_corr:.4f} (p={pearson_p:.2e})")
        print(f"Spearman: {spearman_corr:.4f} (p={spearman_p:.2e})")

    # 4. 시각화
    plt.figure(figsize=(16, 12))

    # 전체 플레이리스트 거리 분포
    plt.subplot(2, 2, 1)
    plt.hist(all_distances, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Average Distance within Playlist")
    plt.ylabel("Frequency")
    plt.title(
        f"All Playlists Distance Distribution\n(n={len(all_distances):,}, mean={all_distances.mean():.4f})"
    )
    plt.grid(True, alpha=0.3)

    # saves가 있는 플레이리스트 분석
    if len(playlist_with_saves) > 0:
        plt.subplot(2, 2, 2)
        plt.scatter(
            playlist_with_saves["saves"],
            playlist_with_saves["avg_distance"],
            alpha=0.6,
            s=15,
        )
        plt.xlabel("Playlist Saves (Followers)")
        plt.ylabel("Average Distance within Playlist")
        plt.title(
            f"Playlist Saves vs Internal Distance\n(n={len(playlist_with_saves):,})"
        )
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.scatter(
            playlist_with_saves["saves"],
            playlist_with_saves["avg_distance"],
            alpha=0.6,
            s=15,
        )
        plt.xlabel("Playlist Saves (Log Scale)")
        plt.ylabel("Average Distance within Playlist")
        plt.xscale("log")
        plt.title("Playlist Saves vs Distance (Log Scale)")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.scatter(
            playlist_with_saves["size"],
            playlist_with_saves["avg_distance"],
            alpha=0.6,
            s=15,
        )
        plt.xlabel("Playlist Size (Track Count)")
        plt.ylabel("Average Distance within Playlist")
        plt.title("Playlist Size vs Internal Distance")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n결과 저장: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()

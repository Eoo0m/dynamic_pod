# benchmark_playlist_similarity.py
# 목적:
#   트랙 임베딩 품질을 벤치마크:
#   - 각 트랙 i 에 대해
#       pos_mean = 같은 플레이리스트(≥1개 공유) 트랙들과의 평균 코사인 유사도
#       neg_mean = 공유하지 않는 트랙들과의 평균 코사인 유사도(샘플 or 전체)
#       margin   = pos_mean - neg_mean
#   - 트랙별 결과 CSV, 전체 통계/히스토그램 저장
#
# 필요 파일:
#   - item2vec_embeddings.npy              (N x dim)
#   - item2vec_meta.parquet OR .csv        (track_id 매핑)
#   - track_playlist_counts.csv            (track_id, playlist_id(파이프 결합), ...)
#
# 실행 예:
#   python benchmark_playlist_similarity.py \
#     --emb item2vec_embeddings.npy --meta item2vec_meta.parquet \
#     --agg track_playlist_counts.csv --neg-mode sample --neg-k 500 --plot
#
# 옵션:
#   --sample-anchors 0      # 0이면 전 트랙 대상, >0이면 앵커 무작위 샘플
#   --pos-cap 0             # 0이면 제한 없음, >0이면 양성 최대 개수 제한
#   --neg-mode all|sample   # 비플리 전부 vs 무작위 샘플
#   --neg-k 500             # neg-mode=sample일 때 음성 샘플 수
#   --seed 42               # 재현성
#
# 출력:
#   embedding_benchmark.csv
#   embedding_benchmark_summary.txt
#   pos_neg_hist.png (옵션 --plot)

import os, argparse, warnings, random
import numpy as np
import pandas as pd

# tqdm (optional)
try:
    from tqdm import tqdm
except Exception:

    class tqdm:
        def __init__(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass


def robust_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
        quotechar='"',
        escapechar="\\",
        sep=",",
    )


def load_embeddings(path="contrastive_top5pct_embeddings.npy"):
    Z = np.load(path)
    if Z.ndim != 2:
        raise SystemExit("embeddings는 2D 배열이어야 합니다.")
    # 코사인 유사도를 위해 L2 정규화
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z


def load_meta(path="contrastive_top5pct_meta.csv"):
    if path.endswith(".parquet"):
        try:
            import pyarrow  # noqa
        except Exception:
            warnings.warn("pyarrow 권장: pip install pyarrow")
        return pd.read_parquet(path)
    return robust_read_csv(path)


def try_load_keys_npy():
    p = "contrastive_top5pct_keys.npy"
    return np.load(p).astype(str).tolist() if os.path.exists(p) else None


def align_meta_to_embeddings(
    meta: pd.DataFrame, Z: np.ndarray
) -> tuple[pd.DataFrame, str]:
    note = ""
    if "track_id" not in meta.columns:
        raise SystemExit("meta에 track_id 컬럼이 필요합니다.")
    keys = try_load_keys_npy()
    if keys is not None:
        m = meta.set_index("track_id").reindex(keys).reset_index()
        miss = int(m["track"].isna().sum()) if "track" in m.columns else 0
        if miss:
            note = f"[경고] keys에는 있으나 meta에 없는 track_id {miss}개."
        return m.fillna(""), note
    if len(meta) == Z.shape[0]:
        note = "[주의] keys.npy 없음 → meta 순서를 임베딩 순서로 가정."
        return meta.reset_index(drop=True), note
    raise SystemExit("meta 행수 ≠ 임베딩 행수. keys.npy 저장을 권장합니다.")


def build_positive_sets(agg_csv: str, valid_tids: set[str]) -> dict[str, set[str]]:
    """
    track_playlist_counts.csv에서 각 트랙의 '동플리 트랙 집합' 구성.
    - 입력 CSV는 track_id별로 playlist_id가 '|'로 결합된 형태.
    - valid_tids: 임베딩/메타에 존재하는 track_id만 사용.
    """
    df = robust_read_csv(agg_csv)
    if "track_id" not in df.columns or "playlist_id" not in df.columns:
        raise SystemExit("집계 CSV에 'track_id'와 'playlist_id' 컬럼이 필요합니다.")
    df = df[df["track_id"].isin(valid_tids)].copy()

    def split_pl(s: str):
        return [p.strip() for p in s.split("|") if p.strip()]

    # playlist_id -> 포함 트랙 목록
    pl2t = {}
    for tid, pls in zip(df["track_id"], df["playlist_id"]):
        for p in split_pl(pls):
            pl2t.setdefault(p, []).append(tid)

    # 각 tid에 대해 같은 플레이리스트 등장 트랙들의 합집합(자기 제외)
    pos = {tid: set() for tid in df["track_id"]}
    for p, tids in pl2t.items():
        uniq = list(dict.fromkeys(tids))
        for t in uniq:
            pos[t].update(x for x in uniq if x != t)
    # valid만 남기기
    for t in list(pos.keys()):
        pos[t] = {x for x in pos[t] if x in valid_tids}
    return pos


def benchmark(
    Z: np.ndarray,
    meta: pd.DataFrame,
    positives: dict[str, set[str]],
    sample_anchors: int = 0,
    pos_cap: int = 0,
    neg_mode: str = "sample",
    neg_k: int = 500,
    seed: int = 42,
):
    rng = random.Random(seed)
    tid_list = meta["track_id"].astype(str).tolist()
    tid2idx = {t: i for i, t in enumerate(tid_list)}
    N = len(tid_list)

    # 앵커 대상: 모든 메타의 track_id 사용 (플레이리스트에 없는 곡도 포함)
    all_track_ids = set(tid_list)
    if sample_anchors and sample_anchors > 0 and sample_anchors < len(all_track_ids):
        anchors = rng.sample(list(all_track_ids), sample_anchors)
    else:
        anchors = list(all_track_ids)

    rows = []
    with tqdm(total=len(anchors), desc="Benchmark", unit="trk") as pbar:
        for tid in anchors:
            i = tid2idx.get(tid)
            if i is None:
                pbar.update(1)
                continue

            pos_ids = list(positives.get(tid, set()))
            # 양성이 없으면 스킵(또는 NaN 기록)
            if not pos_ids:
                rows.append(
                    {
                        "track_id": tid,
                        "track": (
                            meta.iloc[i].get("track", "")
                            if "track" in meta.columns
                            else ""
                        ),
                        "artist": (
                            meta.iloc[i].get("artist", "")
                            if "artist" in meta.columns
                            else ""
                        ),
                        "pos_mean": np.nan,
                        "neg_mean": np.nan,
                        "margin": np.nan,
                        "pos_n": 0,
                        "neg_n": 0,
                    }
                )
                pbar.update(1)
                continue

            if pos_cap and len(pos_ids) > pos_cap:
                pos_ids = rng.sample(pos_ids, pos_cap)

            pos_idx = [tid2idx[x] for x in pos_ids if x in tid2idx]
            pos_idx = [j for j in pos_idx if j != i]
            pos_n = len(pos_idx)

            if pos_n == 0:
                rows.append(
                    {
                        "track_id": tid,
                        "track": (
                            meta.iloc[i].get("track", "")
                            if "track" in meta.columns
                            else ""
                        ),
                        "artist": (
                            meta.iloc[i].get("artist", "")
                            if "artist" in meta.columns
                            else ""
                        ),
                        "pos_mean": np.nan,
                        "neg_mean": np.nan,
                        "margin": np.nan,
                        "pos_n": 0,
                        "neg_n": 0,
                    }
                )
                pbar.update(1)
                continue

            # 음성 후보 = 전체 - {자기자신} - 양성
            pos_set = set(pos_idx)
            neg_candidates = [j for j in range(N) if (j != i and j not in pos_set)]
            if neg_mode == "sample":
                k = min(neg_k, len(neg_candidates))
                neg_idx = rng.sample(neg_candidates, k) if k > 0 else []
            else:
                neg_idx = neg_candidates
            neg_n = len(neg_idx)

            # 코사인 유사도: dot(Z[i], Z[indices])
            zi = Z[i]
            pos_mean = float(np.mean(zi @ Z[pos_idx].T)) if pos_n > 0 else np.nan
            neg_mean = float(np.mean(zi @ Z[neg_idx].T)) if neg_n > 0 else np.nan
            margin = (pos_mean - neg_mean) if (pos_n > 0 and neg_n > 0) else np.nan

            rows.append(
                {
                    "track_id": tid,
                    "track": (
                        meta.iloc[i].get("track", "") if "track" in meta.columns else ""
                    ),
                    "artist": (
                        meta.iloc[i].get("artist", "")
                        if "artist" in meta.columns
                        else ""
                    ),
                    "pos_mean": pos_mean,
                    "neg_mean": neg_mean,
                    "margin": margin,
                    "pos_n": pos_n,
                    "neg_n": neg_n,
                }
            )
            pbar.update(1)

    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame) -> str:
    d = df.dropna(subset=["pos_mean", "neg_mean", "margin"])
    n = len(d)
    if n == 0:
        return "유효한 샘플이 없습니다."
    s = []

    def qq(c):
        return (
            np.mean(c),
            np.std(c),
            np.median(c),
            np.percentile(c, 5),
            np.percentile(c, 25),
            np.percentile(c, 75),
            np.percentile(c, 95),
        )

    pm, psd, pmed, p5, p25, p75, p95 = qq(d["pos_mean"].to_numpy())
    nm, nsd, nmed, n5, n25, n75, n95 = qq(d["neg_mean"].to_numpy())
    mm, msd, mmed, m5, m25, m75, m95 = qq(d["margin"].to_numpy())

    gt = float((d["margin"] > 0).mean())
    s.append(f"[샘플 수] n={n:,}")
    s.append(
        f"[pos_mean] mean={pm:.4f} sd={psd:.4f} median={pmed:.4f} q5={p5:.4f} q25={p25:.4f} q75={p75:.4f} q95={p95:.4f}"
    )
    s.append(
        f"[neg_mean] mean={nm:.4f} sd={nsd:.4f} median={nmed:.4f} q5={n5:.4f} q25={n25:.4f} q75={n75:.4f} q95={n95:.4f}"
    )
    s.append(
        f"[margin]   mean={mm:.4f} sd={msd:.4f} median={mmed:.4f} q5={m5:.4f} q25={m25:.4f} q75={m75:.4f} q95={m95:.4f}"
    )
    s.append(f"[정확도 유사 지표] margin>0 비율 = {gt*100:.2f}%")
    # 간단 t-검정(페어드 아닌 근사)도 참고로
    try:
        from scipy.stats import ttest_rel

        tstat, pval = ttest_rel(d["pos_mean"], d["neg_mean"], nan_policy="omit")
        s.append(f"[t-test(pos vs neg)] t={tstat:.3f}, p={pval:.2e}")
    except Exception:
        s.append("[참고] scipy 없음 → t-test 생략")
    return "\n".join(s)


def summarize_overall_stats(stats: dict) -> str:
    """
    전체 곡 유사도 통계 요약
    """
    s = []
    s.append(
        f"[전체 곡 유사도 통계] 샘플: {stats['sample_size']:,}곡, 쌍: {stats['n_pairs']:,}개"
    )
    s.append(
        f"[전체 유사도] mean={stats['mean']:.4f} std={stats['std']:.4f} median={stats['median']:.4f}"
    )
    s.append(
        f"[전체 유사도] min={stats['min']:.4f} max={stats['max']:.4f} q5={stats['q5']:.4f} q25={stats['q25']:.4f} q75={stats['q75']:.4f} q95={stats['q95']:.4f}"
    )
    return "\n".join(s)


def compute_overall_similarity_stats(
    Z: np.ndarray, sample_size: int = 10000, seed: int = 42
) -> dict:
    """
    전체 곡들 간의 유사도 통계 계산 (샘플링 기반)
    """
    rng = np.random.RandomState(seed)
    N = Z.shape[0]

    # 메모리 절약을 위해 샘플링
    if sample_size > 0 and N > sample_size:
        indices = rng.choice(N, sample_size, replace=False)
        Z_sample = Z[indices]
    else:
        Z_sample = Z

    # 유사도 행렬 계산 (상삼각 부분만)
    n_sample = Z_sample.shape[0]
    similarities = []

    print(f"전체 유사도 통계 계산 중 (샘플 크기: {n_sample:,})...")
    with tqdm(
        total=(n_sample * (n_sample - 1)) // 2, desc="Similarity", unit="pairs"
    ) as pbar:
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                sim = float(np.dot(Z_sample[i], Z_sample[j]))
                similarities.append(sim)
                pbar.update(1)

    similarities = np.array(similarities)

    return {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "median": float(np.median(similarities)),
        "q5": float(np.percentile(similarities, 5)),
        "q25": float(np.percentile(similarities, 25)),
        "q75": float(np.percentile(similarities, 75)),
        "q95": float(np.percentile(similarities, 95)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "n_pairs": len(similarities),
        "sample_size": n_sample,
        "similarities": similarities,
    }


def plot_hist(df: pd.DataFrame, overall_stats: dict = None, path="pos_neg_hist.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[경고] matplotlib 미설치 → 히스토그램 생략")
        return

    d = df.dropna(subset=["pos_mean", "neg_mean", "margin"])

    # 전체 유사도 포함 시 4개 서브플롯, 아니면 3개
    n_plots = 4 if overall_stats else 3
    plt.figure(figsize=(12, 8))

    plt.subplot(n_plots, 1, 1)
    plt.hist(d["pos_mean"], bins=40, alpha=0.7, color="green")
    plt.title("Positive Mean Similarity (같은 플레이리스트 곡들)")
    plt.xlabel("Cosine Similarity")

    plt.subplot(n_plots, 1, 2)
    plt.hist(d["neg_mean"], bins=40, alpha=0.7, color="red")
    plt.title("Negative Mean Similarity (다른 플레이리스트 곡들)")
    plt.xlabel("Cosine Similarity")

    plt.subplot(n_plots, 1, 3)
    plt.hist(d["margin"], bins=40, alpha=0.7, color="blue")
    plt.title("Margin = Positive - Negative")
    plt.xlabel("Similarity Difference")

    if overall_stats:
        plt.subplot(n_plots, 1, 4)
        plt.hist(overall_stats["similarities"], bins=50, alpha=0.7, color="purple")
        plt.title(
            f"Overall Track Similarity Distribution (샘플: {overall_stats['sample_size']:,}곡)"
        )
        plt.xlabel("Cosine Similarity")
        plt.axvline(
            overall_stats["mean"],
            color="black",
            linestyle="--",
            label=f"Mean: {overall_stats['mean']:.4f}",
        )
        plt.axvline(
            overall_stats["median"],
            color="orange",
            linestyle="--",
            label=f"Median: {overall_stats['median']:.4f}",
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[저장] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="contrastive_top5pct_embeddings.npy")
    ap.add_argument("--meta", default="contrastive_top5pct_meta.csv")
    ap.add_argument("--agg", default="preprocessing/track_playlist_counts_top5pct.csv")
    ap.add_argument("--sample-anchors", type=int, default=0)
    ap.add_argument("--pos-cap", type=int, default=0)
    ap.add_argument("--neg-mode", choices=["all", "sample"], default="sample")
    ap.add_argument("--neg-k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--overall-sample",
        type=int,
        default=5000,
        help="전체 유사도 계산을 위한 샘플 크기 (0=전체)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    Z = load_embeddings(args.emb)
    meta_raw = load_meta(args.meta)
    meta, note = align_meta_to_embeddings(meta_raw, Z)
    if note:
        print(note)

    valid_tids = set(meta["track_id"].astype(str).tolist())
    positives = build_positive_sets(args.agg, valid_tids)

    df = benchmark(
        Z,
        meta,
        positives,
        sample_anchors=args.sample_anchors,
        pos_cap=args.pos_cap,
        neg_mode=args.neg_mode,
        neg_k=args.neg_k,
        seed=args.seed,
    )
    df.to_csv("embedding_benchmark.csv", index=False)
    print(f"[저장] embedding_benchmark.csv ({len(df):,} rows)")

    summ = summarize(df)

    # 전체 유사도 통계 계산
    overall_stats = compute_overall_similarity_stats(
        Z, sample_size=args.overall_sample, seed=args.seed
    )
    overall_summ = summarize_overall_stats(overall_stats)

    full_summary = summ + "\n\n" + overall_summ

    with open("embedding_benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write(full_summary + "\n")
    print("[요약]\n" + full_summary)

    if args.plot:
        plot_hist(df, overall_stats)


if __name__ == "__main__":
    main()

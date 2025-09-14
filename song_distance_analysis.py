# poscount_sim_stats.py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import argparse


# --- 재사용: 네가 이미 가진 로더 ---
def load_song_data(prefix: str):
    Z = np.load(f"{prefix}_embeddings.npy")  # (N,d)
    keys = np.load(f"{prefix}_keys.npy")  # (N,)
    meta = pd.read_csv(f"{prefix}_meta.csv")  # track, artist, album, pos_count 등
    return Z, keys, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="contrastive_learning/contrastive_top5pct")
    ap.add_argument("--sample", type=int, default=2000, help="샘플 곡 개수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--save-csv", default="", help="결과 CSV 저장 경로(옵션)")
    ap.add_argument("--renorm", action="store_true", help="임베딩 재정규화")
    args = ap.parse_args()

    Z, keys, meta = load_song_data(args.prefix)
    N, d = Z.shape
    print(f"[INFO] embeddings={Z.shape}, tracks={len(keys)}")

    if args.renorm:
        Z = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)

    # pos_count 정리
    if "pos_count" not in meta.columns:
        raise SystemExit("meta에 'pos_count' 컬럼이 필요합니다.")
    meta["pos_count"] = (
        pd.to_numeric(meta["pos_count"], errors="coerce").fillna(0).astype(int)
    )

    # 인덱싱 빠르게
    key2idx = {k: i for i, k in enumerate(keys)}
    if "track_id" not in meta.columns:
        raise SystemExit("meta에 'track_id' 컬럼이 필요합니다.")

    # 샘플링 풀 (원하면 pos_count>0로 제한 가능)
    pool = meta["track_id"].tolist()
    rng = np.random.default_rng(args.seed)
    sample_ids = rng.choice(pool, size=min(args.sample, len(pool)), replace=False)
    idxs = np.array([key2idx[t] for t in sample_ids])

    # 배치 계산
    B = args.batch
    mean_sims = np.empty(len(idxs), dtype=np.float32)
    var_sims = np.empty(len(idxs), dtype=np.float32)

    for s in tqdm(range(0, len(idxs), B), desc="Compute mean/var(sim)"):
        e = min(len(idxs), s + B)
        batch_idx = idxs[s:e]
        sims = Z[batch_idx] @ Z.T  # (b, N) 코사인 유사도(정규화 가정)
        sims[np.arange(e - s), batch_idx] = np.nan  # 자기 자신 제외
        mean_sims[s:e] = np.nanmean(sims, axis=1)
        var_sims[s:e] = np.nanvar(sims, axis=1)

    # 결과 DF
    out = pd.DataFrame(
        {
            "track_id": sample_ids,
            "idx": idxs,
            "pos_count": meta.set_index("track_id").loc[sample_ids, "pos_count"].values,
            "mean_sim": mean_sims,
            "var_sim": var_sims,
        }
    )

    # 상관관계
    pear_m = pearsonr(out["pos_count"], out["mean_sim"])
    spear_m = spearmanr(out["pos_count"], out["mean_sim"])
    pear_v = pearsonr(out["pos_count"], out["var_sim"])
    spear_v = spearmanr(out["pos_count"], out["var_sim"])

    print("\n=== 상관관계(샘플) ===")
    print(
        f"pos_count ↔ mean_sim : Pearson r={pear_m.statistic:.4f} (p={pear_m.pvalue:.2e}), "
        f"Spearman ρ={spear_m.statistic:.4f} (p={spear_m.pvalue:.2e})"
    )
    print(
        f"pos_count ↔ var_sim  : Pearson r={pear_v.statistic:.4f} (p={pear_v.pvalue:.2e}), "
        f"Spearman ρ={spear_v.statistic:.4f} (p={spear_v.pvalue:.2e})"
    )

    # 분위(디사일)별 요약
    out["decile"] = pd.qcut(out["pos_count"].rank(method="first"), 10, labels=False) + 1
    dec = (
        out.groupby("decile")
        .agg(
            pos_count_mean=("pos_count", "mean"),
            mean_sim_mean=("mean_sim", "mean"),
            mean_sim_med=("mean_sim", "median"),
            var_sim_mean=("var_sim", "mean"),
            var_sim_med=("var_sim", "median"),
            n=("track_id", "count"),
        )
        .round(4)
    )
    print("\n=== pos_count 디사일별 요약 ===")
    print(dec)

    if args.save_csv:
        out.to_csv(args.save_csv, index=False)
        print(f"[저장] {args.save_csv}  rows={len(out)}")


if __name__ == "__main__":
    main()

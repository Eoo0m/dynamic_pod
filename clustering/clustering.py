import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import umap
import matplotlib.pyplot as plt


# ---------- 로딩/정렬 ----------
def load_embeddings(path="contrastive_learning/contrastive_top5pct_win_embeddings.npy"):
    Z = np.load(path)
    if Z.ndim != 2:
        raise SystemExit("embeddings는 2D여야 합니다.")
    return Z


def load_meta(path="contrastive_learning/contrastive_top5pct_win_meta.csv"):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def align_meta_to_embeddings(meta: pd.DataFrame, Z: np.ndarray):
    if len(meta) == Z.shape[0]:
        return (
            meta.reset_index(drop=True),
            "[주의] keys 없음 → meta 순서를 임베딩 순서로 가정.",
        )
    raise SystemExit("meta 행수 ≠ 임베딩 행수. keys.npy 저장을 권장.")


# ---------- 전처리 ----------
def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def pca_reduce(Z, n=50):
    if n and Z.shape[1] > n:
        return PCA(n_components=n, random_state=42).fit_transform(Z)
    return Z


def umap_reduce(Z, n=15, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_components=n, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
    )
    return reducer.fit_transform(Z)


# ---------- 필터(허브/이상치) ----------
def filter_by_num_playlists(meta, q_hi=0.995):
    if "pos_count" not in meta.columns:
        print("[경고] 'pos_count' 컬럼이 없습니다. 기본값으로 0으로 처리합니다.")
        meta["pos_count"] = 0
    v = pd.to_numeric(meta["pos_count"], errors="coerce").fillna(0).to_numpy()
    thr = float(np.quantile(v, q_hi))
    return (v <= thr), thr


def hubness_scores(Zn, k=50, batch=2048):
    n = Zn.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        sims = Zn[s:e] @ Zn.T
        for i in range(e - s):
            sims[i, s + i] = -1.0
        top = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        sc = np.take_along_axis(sims, top, axis=1).mean(axis=1)
        scores[s:e] = sc
    return scores


def filter_by_hubness(Zn, q_hi=0.995, k=50):
    hs = hubness_scores(Zn, k=k)
    thr = float(np.quantile(hs, q_hi))
    return (hs <= thr), thr


def filter_by_isoforest(Zr, contamination=0.02):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    ).fit(Zr)
    return (clf.predict(Zr) == 1), contamination


# ---------- 클러스터링 ----------
def cluster_hdbscan(
    Zr,
    min_cluster=5,
    min_samples=None,
    selection="eom",
    epsilon=0.0,
    metric="euclidean",
):
    clt = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=min_samples or min_cluster,
        metric=metric,
        cluster_selection_method=selection,
        cluster_selection_epsilon=float(epsilon),
        core_dist_n_jobs=0,
    )
    return clt.fit_predict(Zr)


def cluster_kmeans(Zr, k=50):
    km = KMeans(n_clusters=int(k), n_init="auto", random_state=42)
    return km.fit_predict(Zr)


# ---------- 요약/플롯 ----------
def summarize(meta_kept, labels, topn=20):
    df = meta_kept.copy()
    df["cluster"] = labels
    use = df[df["cluster"] != -1]
    if "pos_count" in use.columns:
        use = use.copy()
        use["pos_count_num"] = pd.to_numeric(use["pos_count"], errors="coerce").fillna(
            0
        )
    rows = []
    for cid, g in use.groupby("cluster"):
        if "pos_count_num" in g.columns:
            g_sorted = g.sort_values("pos_count_num", ascending=False)
            top_artists = g_sorted["artist"].drop_duplicates().head(topn).tolist()
            top_tracks = g_sorted["track"].drop_duplicates().head(topn).tolist()
        else:
            top_artists = g["artist"].value_counts().head(topn).index.tolist()
            top_tracks = g["track"].value_counts().head(topn).index.tolist()
        rows.append(
            {
                "cluster": int(cid),
                "size": int(len(g)),
                "top_artists": " | ".join(top_artists),
                "top_tracks": " | ".join(top_tracks),
            }
        )
    return df, pd.DataFrame(rows).sort_values("size", ascending=False)


def plot_2d(Zr, labels, meta_kept, path="clustering/clusters_2d.png", dpi=140):
    X2 = PCA(n_components=2, random_state=42).fit_transform(Zr)
    plt.figure(figsize=(12, 9))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    m = labels == -1
    if m.any():
        plt.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.3, color="gray", label="noise(-1)")
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_points = X2[mask]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=12,
            alpha=0.7,
            color=colors[i],
            label=f"Cluster {label}",
        )
        center = cluster_points.mean(axis=0)
        cluster_meta = meta_kept[mask]
        if "pos_count" in cluster_meta.columns:
            cluster_meta = cluster_meta.copy()
            cluster_meta["pos_count_num"] = pd.to_numeric(
                cluster_meta["pos_count"], errors="coerce"
            ).fillna(0)
            top_tracks_data = cluster_meta.nlargest(2, "pos_count_num")[
                ["track", "artist"]
            ].values
        else:
            top_tracks_data = cluster_meta[["track", "artist"]].head(2).values
        if len(top_tracks_data) > 0:
            track_text = " | ".join([f"{t} - {a}" for t, a in top_tracks_data])
            plt.annotate(
                f"C{label}: {track_text}",
                xy=center,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3),
            )
    plt.title("Cluster Visualization with Representative Tracks", fontsize=14)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[저장] {path}")
    plt.show()


# ---------- 자동 선택(핵심) ----------
def score_labels(labels, target_k=100, max_noise=0.65):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return -1e9, n_clusters, 1.0
    noise = float((labels == -1).mean())
    # 점수: target_k에 가까울수록↑, 노이즈 낮을수록↑
    score = -abs(n_clusters - target_k) * 3.0 - max(0.0, noise - max_noise) * 200.0
    return score, n_clusters, noise


def auto_hdbscan(Z_list, args):
    """
    Z_list: [(name, Z2d), ...]  # PCA, UMAP 등 후보 임베딩 목록
    여러 파라미터 조합을 시도하여 score가 높은 것을 선택.
    """
    grid_metrics = ["cosine", "euclidean"]
    grid_selection = ["leaf", "eom"]
    grid_mc_ms = [(5, 3), (10, 5), (15, 5), (20, 10), (30, 10)]
    best = None

    for emb_name, Zr in Z_list:
        for metric in grid_metrics:
            for sel in grid_selection:
                for mc, ms in grid_mc_ms:
                    try:
                        labels = cluster_hdbscan(
                            Zr,
                            min_cluster=mc,
                            min_samples=ms,
                            selection=sel,
                            epsilon=args.epsilon,
                            metric=metric,
                        )
                    except Exception as e:
                        continue
                    s, nk, noise = score_labels(
                        labels, target_k=args.target_k, max_noise=args.auto_max_noise
                    )
                    cand = (s, nk, noise, emb_name, metric, sel, mc, ms, labels)
                    if (best is None) or (cand[0] > best[0]):
                        best = cand

    if best is None:
        raise SystemExit("auto_hdbscan 실패: 시도한 조합이 모두 에러.")
    s, nk, noise, emb_name, metric, sel, mc, ms, labels = best
    print(
        f"[auto] emb={emb_name}, metric={metric}, sel={sel}, min_cluster={mc}, min_samples={ms} "
        f"→ clusters={nk}, noise={noise:.2f}, score={s:.1f}"
    )
    return labels


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--emb", default="contrastive_learning/contrastive_top5pct_win_embeddings.npy"
    )
    ap.add_argument(
        "--meta", default="contrastive_learning/contrastive_top5pct_win_meta.csv"
    )
    ap.add_argument("--algo", choices=["hdbscan", "kmeans"], default="hdbscan")
    ap.add_argument("--pca", type=int, default=50)

    # HDBSCAN 기본
    ap.add_argument("--min-cluster", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--selection", choices=["eom", "leaf"], default="eom")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")

    # KMeans
    ap.add_argument("--k", type=int, default=50)

    # 필터
    ap.add_argument("--qnum-hi", type=float, default=0.995)
    ap.add_argument("--hub-qhi", type=float, default=0.995)
    ap.add_argument("--hub-k", type=int, default=50)
    ap.add_argument("--if-contam", type=float, default=0.02)

    # 임베딩(UMAP)
    ap.add_argument("--umap", type=int, default=0, help="UMAP 차원(0=미사용)")
    ap.add_argument("--umap-neigh", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)

    # 자동 모드
    ap.add_argument(
        "--auto", action="store_true", default=True, help="HDBSCAN 자동 스윕 사용"
    )
    ap.add_argument(
        "--target-k", type=int, default=100, help="목표 군집 수(자동 선택 기준)"
    )
    ap.add_argument(
        "--auto-max-noise",
        type=float,
        default=0.65,
        help="허용 노이즈 비율 상한(페널티 기준)",
    )

    # 저장/플롯
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--outdir", default="clustering", help="결과 저장 폴더(기본: clustering/)"
    )
    ap.add_argument("--dpi", type=int, default=140, help="플롯 저장 DPI")

    args = ap.parse_args()

    # 로드
    Z = load_embeddings(args.emb)
    m0 = load_meta(args.meta)
    if "track_id" not in m0.columns:
        raise SystemExit("meta에 track_id 필요")
    meta, note = align_meta_to_embeddings(m0, Z)
    if note:
        print(note)

    # 저장 디렉토리
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 필터링 마스크
    keep = np.ones(len(meta), bool)
    if args.qnum_hi > 0:
        k1, thr = filter_by_num_playlists(meta, q_hi=args.qnum_hi)
        keep &= k1
        print(
            f"[num_playlists] keep {keep.sum()}/{len(keep)} (<= q{args.qnum_hi} thr={thr:.1f})"
        )

    # PCA → 정규화
    Zp = pca_reduce(Z, n=args.pca)
    Zn = l2norm(Zp)

    if args.hub_qhi > 0:
        k2_small, thr = filter_by_hubness(Zn[keep], q_hi=args.hub_qhi, k=args.hub_k)
        idx = np.where(keep)[0]
        tmp = np.zeros_like(keep, bool)
        tmp[idx[k2_small]] = True
        keep &= tmp
        print(
            f"[hubness]      keep {keep.sum()}/{len(tmp)} (<= q{args.hub_qhi} thr={thr:.4f})"
        )

    if args.if_contam > 0:
        k3_small, _ = filter_by_isoforest(Zn[keep], contamination=args.if_contam)
        idx = np.where(keep)[0]
        tmp = np.zeros_like(keep, bool)
        tmp[idx[k3_small]] = True
        keep &= tmp
        print(f"[IForest]      keep {keep.sum()} rows")

    # 임베딩 후보 준비(PCA, UMAP)
    Z_used_list = [("pca", l2norm(Zn[keep]))]
    if args.umap and args.umap > 0:
        Zu = umap_reduce(
            Zn, n=args.umap, n_neighbors=args.umap_neigh, min_dist=args.umap_min_dist
        )
        Z_used_list.append(("umap", l2norm(Zu[keep])))

    # 대상 행렬/메타
    # (자동모드에서 여러 후보를 돌리므로 keep만 지금 적용)
    meta_kept = meta.loc[keep].reset_index(drop=True)
    print(
        f"[최종] 대상 {len(meta_kept)} rows, dim_candidates={[Zc.shape[1] for _, Zc in Z_used_list]}"
    )

    # --- 클러스터링 ---
    if args.algo == "hdbscan":
        if args.auto:
            labels = auto_hdbscan(Z_used_list, args)
        else:
            # 수동
            Z_final = Z_used_list[-1][1]  # 마지막 후보(UMAP 있으면 UMAP, 없으면 PCA)
            labels = cluster_hdbscan(
                Z_final,
                min_cluster=args.min_cluster,
                min_samples=args.min_samples,
                selection=args.selection,
                epsilon=args.epsilon,
                metric=args.metric,
            )

        # 폴백: 군집 너무 적으면 KMeans
        n_k = len(set(labels)) - (1 if -1 in labels else 0)
        if n_k <= 2:
            n = len(meta_kept)
            k_auto = int(np.clip(int(np.sqrt(n / 2)), 20, 200))
            print(f"[fallback] HDBSCAN clusters={n_k} → KMeans(k={k_auto})로 폴백")
            Z_final = Z_used_list[-1][1]
            labels = cluster_kmeans(Z_final, k=k_auto)
    else:
        Z_final = Z_used_list[-1][1]
        labels = cluster_kmeans(Z_final, k=args.k)

    # 요약/저장
    # (Z_final은 마지막 사용 임베딩으로 설정)
    if "Z_final" not in locals():
        Z_final = Z_used_list[-1][1]
    df, summ = summarize(meta_kept, labels)

    clusters_csv = os.path.join(outdir, "clusters.csv")
    summary_csv = os.path.join(outdir, "cluster_summary.csv")
    df.to_csv(clusters_csv, index=False)
    summ.to_csv(summary_csv, index=False)
    print(
        f"[저장] {clusters_csv} ({len(df):,}) / {summary_csv} (clusters={len(summ):,})"
    )

    if args.plot:
        plot_2d(
            Z_final,
            labels,
            meta_kept,
            path=os.path.join(outdir, "clusters_2d.png"),
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

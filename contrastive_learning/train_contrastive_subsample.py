# train_contrastive_top1pct.py  (neighbors=positives, pos_count := len(neighbors))
import argparse, random, math, os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


# ---------- 로드/전처리 ----------
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


def split_ids(s: str):
    return [p.strip() for p in (s or "").split("|") if p and p.strip()]


def build_graph(df: pd.DataFrame):
    """
    neighbors 컬럼을 이용해 track 간 양성 그래프 구성.
    pos_count는 'neighbors' 길이로만 계산한다.
    """
    need = {"track_id", "track", "artist", "album", "neighbors"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"필수 컬럼 없음: {miss}")

    # 메타/ID
    meta = (
        df[["track_id", "track", "artist", "album"]].drop_duplicates("track_id").copy()
    )
    tids = meta["track_id"].tolist()
    meta.index = meta["track_id"]
    tid_set = set(tids)

    # neighbors -> 양성 집합(양방향 보장)
    pos_sets = defaultdict(set)
    for tid, neigh_str in zip(df["track_id"], df["neighbors"]):
        if tid not in tid_set:
            continue
        for nb in split_ids(neigh_str):
            if nb == tid or nb not in tid_set:
                continue
            pos_sets[tid].add(nb)
            pos_sets[nb].add(tid)

    # 무조건 neighbors 길이로 pos_count 정의
    pos_count = {t: len(pos_sets.get(t, set())) for t in tids}
    return meta, tids, pos_sets, pos_count


# ---------- 배치 샘플러 ----------
class PositiveAwareBatcher:
    """앵커 + 해당 양성(1~2개)을 같은 배치에 섞음."""

    def __init__(self, anchors, pos_sets, batch_size=256, fill_pool=None, seed=42):
        self.anchors = anchors
        self.pos_sets = pos_sets
        self.batch = batch_size
        self.fill_pool = fill_pool or anchors
        self.rng = random.Random(seed)

    def __iter__(self):
        A = self.anchors[:]
        self.rng.shuffle(A)
        cur = []
        for a in A:
            cur.append(a)
            cand = list(self.pos_sets.get(a, []))
            if cand:
                self.rng.shuffle(cand)
                for x in cand[: self.rng.choice([1, 2])]:
                    cur.append(x)
            if len(cur) >= self.batch:
                uniq = list(dict.fromkeys(cur))
                if len(uniq) >= self.batch:
                    self.rng.shuffle(uniq)
                    yield uniq[: self.batch]
                else:
                    need = self.batch - len(uniq)
                    fill = self.rng.sample(
                        self.fill_pool, k=min(need, len(self.fill_pool))
                    )
                    uniq.extend(fill[:need])
                    yield uniq
                cur = []
        if cur:
            uniq = list(dict.fromkeys(cur))
            if len(uniq) >= self.batch:
                self.rng.shuffle(uniq)
                yield uniq[: self.batch]
            else:
                need = self.batch - len(uniq)
                fill = self.rng.sample(self.fill_pool, k=min(need, len(self.fill_pool)))
                uniq.extend(fill[:need])
                yield uniq


# ---------- 모델 ----------
class TrackTable(nn.Module):
    def __init__(self, n_items: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        z = self.emb(idx)
        return nn.functional.normalize(z, dim=1)


# ---------- 유틸 ----------
def build_pos_mask_fast(batch_ids, pos_sets):
    """배치 내 이웃만 순회해 BxB 양성 마스크 생성 (O(∑deg))."""
    B = len(batch_ids)
    mask = torch.zeros((B, B), dtype=torch.bool)
    in_batch = {tid: i for i, tid in enumerate(batch_ids)}
    for i, ti in enumerate(batch_ids):
        for nb in pos_sets.get(ti, ()):
            j = in_batch.get(nb)
            if j is not None and j != i:
                mask[i, j] = True
    return mask


def subsample_pos_mask(pos_mask: torch.Tensor, P: int, rng: np.random.Generator):
    B = pos_mask.size(0)
    out = torch.zeros_like(pos_mask)
    for i in range(B):
        idx = torch.nonzero(pos_mask[i], as_tuple=False).flatten().cpu().numpy()
        if len(idx) == 0:
            continue
        choose = rng.choice(idx, size=min(P, len(idx)), replace=False)
        out[i, choose] = True
    return out


def mp_infonce_loss(z, tau, pos_mask, pos_sel_mask, anchor_weight=None):
    B = z.size(0)
    sim = (z @ z.t()) / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -float("inf"))

    full_pos_mask = pos_mask.to(z.device)
    pos_logits = sim.masked_fill(~pos_sel_mask.to(z.device), -float("inf"))

    neg_mask = (~full_pos_mask) & (~eye)
    denom_mask = neg_mask | pos_sel_mask.to(z.device)
    denom_logits = sim.masked_fill(~denom_mask, -float("inf"))

    has_pos = (pos_sel_mask.any(dim=1)).to(z.device)
    if has_pos.sum() == 0:
        return sim.new_tensor(0.0), 0

    pos_lse = torch.logsumexp(pos_logits[has_pos], dim=1)
    all_lse = torch.logsumexp(denom_logits[has_pos], dim=1)
    loss = -(pos_lse - all_lse)

    if anchor_weight is not None:
        w = anchor_weight[has_pos].to(z.device)
        loss = loss * w
    return loss.mean(), int(has_pos.sum().item())


# ---------- 학습 루프 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="preprocessing/track_playlist_counts_top5pct_win10.csv",
        help="입력 CSV (neighbors 포함)",
    )
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.06)
    ap.add_argument("--pos-per-anchor", type=int, default=5)
    ap.add_argument("--min-pos-anchor", type=int, default=0)
    ap.add_argument("--hub-cap", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-prefix", default="contrastive_top5pct_win")
    ap.add_argument(
        "--anchors-per-epoch",
        type=int,
        default=0,
        help="에폭마다 사용할 앵커 수 상한(0이면 전체 사용)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.csv):
        raise SystemExit(f"입력 CSV가 없습니다: {args.csv}")

    df = robust_read_csv(args.csv)
    meta, tids, pos_sets, pos_count = build_graph(df)
    N = len(tids)
    tid2idx = {t: i for i, t in enumerate(tids)}

    # 앵커: neighbors 수 기준
    anchors = [t for t in tids if pos_count.get(t, 0) >= args.min_pos_anchor]
    anchors_full = (
        anchors
        if (args.anchors_per_epoch and args.anchors_per_epoch < len(anchors))
        else None
    )
    print(
        f"[INFO] tracks={N:,}, anchors={len(anchors):,}, min_pos_anchor={args.min_pos_anchor}"
    )

    # 허브 역가중 (허브일수록 작게), 평균≈1 정규화
    cap = max(1, args.hub_cap)
    counts = np.array([max(pos_count.get(t, 0), 1) for t in tids], dtype=np.float32)
    aw = (cap / np.maximum(counts, cap)) ** 0.5  # β=0.5~1 가능
    aw = aw * (len(aw) / aw.sum())
    anchor_weight = torch.from_numpy(aw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackTable(N, args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    def make_batcher(ep):
        if anchors_full is not None:
            A = anchors_full[:]
            random.Random(args.seed + ep).shuffle(A)
            A = A[: args.anchors_per_epoch]
        else:
            A = anchors
        steps_per_epoch = max(1, math.ceil(len(A) / args.batch))
        batcher = PositiveAwareBatcher(
            anchors=A,
            pos_sets=pos_sets,
            batch_size=args.batch,
            fill_pool=tids,
            seed=args.seed + ep,
        )
        return batcher, steps_per_epoch

    for ep in range(1, args.epochs + 1):
        model.train()
        batcher, steps_per_epoch = make_batcher(ep)
        pbar = tqdm(batcher, total=steps_per_epoch, desc=f"Epoch {ep}/{args.epochs}")
        total_loss, total_anchor, iters = 0.0, 0, 0

        for batch_ids in pbar:
            idx = torch.tensor(
                [tid2idx[t] for t in batch_ids], dtype=torch.long, device=device
            )
            z = model(idx)

            pos_mask = build_pos_mask_fast(batch_ids, pos_sets).to(device)
            pos_sel_mask = subsample_pos_mask(
                pos_mask.detach().cpu(), args.pos_per_anchor, rng
            ).to(device)
            aw_batch = anchor_weight[idx.detach().cpu()].to(device)

            loss, used = mp_infonce_loss(z, args.tau, pos_mask, pos_sel_mask, aw_batch)
            if used == 0:
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_anchor += used
            iters += 1
            pbar.set_postfix(
                loss=f"{total_loss/max(1,iters):.4f}", anchors_used=total_anchor
            )

        print(
            f"[Epoch {ep}] mean_loss={total_loss/max(1,iters):.4f}, anchors_used={total_anchor}, steps={iters}"
        )

    # 저장 (L2 정규화된 임베딩)
    with torch.no_grad():
        all_idx = torch.arange(N, dtype=torch.long, device=device)
        Z = model(all_idx).detach().cpu().numpy()

    base_dir = "contrastive_learning"
    os.makedirs(base_dir, exist_ok=True)
    prefix = (
        args.save_prefix
        if os.path.dirname(args.save_prefix)
        else os.path.join(base_dir, args.save_prefix)
    )
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    np.save(f"{prefix}_embeddings.npy", Z)
    np.save(f"{prefix}_keys.npy", np.array(tids))
    meta.assign(pos_count=[pos_count.get(t, 0) for t in meta.index]).to_csv(
        f"{prefix}_meta.csv", index=False
    )

    print(f"[저장] {prefix}_embeddings.npy  shape={Z.shape}")
    print(f"[저장] {prefix}_keys.npy       rows={len(tids)}")
    print(f"[저장] {prefix}_meta.csv       rows={len(meta)}")


if __name__ == "__main__":
    main()

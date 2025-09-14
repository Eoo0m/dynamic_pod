# filter_top5pct_by_neighbors.py
import sys, argparse
import pandas as pd
import numpy as np


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


def calc_percentile_threshold(series: pd.Series, pct: float = 95.0) -> int:
    """상위 (100-pct)% 선별용 임계값. pct=95 → 상위 5%"""
    c = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if c.size == 0:
        return 0
    try:
        q = np.percentile(c, pct, method="nearest")  # numpy >= 1.22
    except TypeError:
        q = np.percentile(c, pct, interpolation="nearest")  # numpy < 1.22
    return int(q)


def neighbors_len(s: str) -> int:
    if not isinstance(s, str) or not s:
        return 0
    return sum(1 for x in s.split("|") if x.strip())


def main():
    ap = argparse.ArgumentParser(
        description="neighbors(윈도우 이웃 수) 기준 상위 5% 필터"
    )
    ap.add_argument(
        "in_csv",
        nargs="?",
        default="preprocessing/track_playlist_counts_with_window.csv",
    )
    ap.add_argument(
        "out_csv", nargs="?", default="preprocessing/track_playlist_counts_top5pct.csv"
    )
    ap.add_argument("--min", type=int, default=None, help="임계값 직접 지정")
    ap.add_argument("--pct", type=float, default=95.0, help="분위수(기본=95 → 상위 5%)")
    args = ap.parse_args()

    df = robust_read_csv(args.in_csv)

    # 기준 시리즈 결정: pos_count_win(있으면) → neighbors 길이(없으면)
    if "pos_count_win" in df.columns:
        basis_col = "pos_count_win"
        basis_series = pd.to_numeric(df[basis_col], errors="coerce").fillna(0)
    elif "neighbors" in df.columns:
        basis_col = "pos_count_win"  # 없던 경우 계산해서 이 이름으로 보존
        df[basis_col] = df["neighbors"].apply(neighbors_len)
        basis_series = pd.to_numeric(df[basis_col], errors="coerce").fillna(0)
    else:
        raise SystemExit("CSV에 'pos_count_win'도 'neighbors'도 없습니다.")

    # 임계값 계산(직접 지정 우선)
    thr = (
        int(args.min)
        if args.min is not None
        else calc_percentile_threshold(basis_series, args.pct)
    )

    cols = list(df.columns)  # 기존/추가 컬럼 전체 보존
    mask = basis_series >= thr
    top = df[mask].loc[:, cols]

    print(
        f"rows={len(df):,}, basis={basis_col}, threshold(pct={args.pct:.1f}, {basis_col} ≥ {thr}), selected={len(top):,}"
    )
    top.to_csv(args.out_csv, index=False)
    print(f"[저장] {args.out_csv}")


if __name__ == "__main__":
    main()

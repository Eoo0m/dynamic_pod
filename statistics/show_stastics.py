# count_stats.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN = sys.argv[1] if len(sys.argv) > 1 else "track_playlist_counts.csv"
OUT_TXT = sys.argv[2] if len(sys.argv) > 2 else "count_stats.txt"
OUT_PNG = sys.argv[3] if len(sys.argv) > 3 else "count_histogram.png"


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


def main():
    df = robust_read_csv(IN)

    if "count" not in df.columns:
        raise SystemExit("CSV에 'count' 컬럼이 없습니다. (집계 결과 파일을 넣어주세요)")

    # 숫자화
    c = pd.to_numeric(df["count"], errors="coerce").dropna()
    c = c.astype(np.int64)

    # 통계
    n = int(c.shape[0])
    mean = float(c.mean())
    std = float(c.std(ddof=1))  # 표본 표준편차
    med = float(c.median())
    mn = int(c.min()) if n else 0
    mx = int(c.max()) if n else 0
    p90 = float(c.quantile(0.90))
    p95 = float(c.quantile(0.95))
    p99 = float(c.quantile(0.99))

    # 텍스트 저장
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"n={n}\n")
        f.write(f"mean={mean:.4f}\n")
        f.write(f"std={std:.4f}\n")
        f.write(f"median={med:.4f}\n")
        f.write(f"min={mn}\nmax={mx}\n")
        f.write(f"p90={p90:.4f}\np95={p95:.4f}\np99={p99:.4f}\n")
    print(f"[저장] {OUT_TXT}")

    # 히스토그램 (단일 플롯, 색 지정 안 함)
    plt.figure(figsize=(8, 5))
    bins = "auto"  # 데이터 크기에 따라 자동
    plt.hist(c.values, bins=bins)
    plt.title("Distribution of 'count'")
    plt.xlabel("count (tracks' playlist occurrence)")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"[저장] {OUT_PNG}")


if __name__ == "__main__":
    main()

# analyze_playlist_saves.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (선택) SciPy의 find_peaks 사용. 없으면 간단한 대체 로직 사용.
try:
    from scipy.signal import find_peaks

    USE_SCIPY = True
except Exception:
    USE_SCIPY = False

CSV_PATH = "crawler/playlist_tracks.csv"  # 경로/파일명 맞게 수정


def nice_row(series: pd.Series) -> str:
    """Series(통계) → 가로 한 줄, 정수 표시"""
    vals = [f"{int(v):d}" for v in series.fillna(0).to_list()]
    return "  ".join([f"{k:>12s}={v:>8s}" for k, v in zip(series.index, vals)])


def quantiles(
    s: pd.Series, qs=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
) -> pd.Series:
    qv = s.quantile(list(qs))
    qv.index = [f"p{int(q*100):02d}" for q in qs]
    return qv


def summarize(values: pd.Series) -> pd.Series:
    """정수형 통계 요약(가로 출력용)"""
    out = pd.Series(dtype="float64")
    out["count"] = values.size
    out["mean"] = values.mean()
    out["var"] = values.var()
    out["std"] = values.std()
    out["min"] = values.min()
    out = pd.concat([out, quantiles(values)], axis=0)
    out["max"] = values.max()
    # 정수 변환
    return out.fillna(0).astype(int)


def kde_peaks_0_1000(data_0_1000: pd.Series, bw_method=None):
    """
    Pandas KDE로 x,y를 얻고, 극점 탐지.
    SciPy가 있으면 find_peaks, 없으면 간단 로컬 최대치로 대체.
    반환: (x, y, peaks_idx, peak_points)
    """
    ax = data_0_1000.plot(kind="kde", bw_method=bw_method)
    line = ax.get_lines()[0]
    x = line.get_xdata()
    y = line.get_ydata()
    plt.close(ax.figure)  # 임시 플롯 닫기

    # 0~1000 범위만 남기기(보정)
    mask = (x >= 0) & (x <= 1000)
    x = x[mask]
    y = y[mask]

    if USE_SCIPY:
        peaks_idx, _ = find_peaks(y)
    else:
        # 간단 대체: 양 옆보다 큰 지점
        peaks_idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1

    peak_points = list(zip(x[peaks_idx], y[peaks_idx]))
    return x, y, peaks_idx, peak_points


def main():
    # 1) 로드
    df = pd.read_csv(CSV_PATH)

    # 2) playlist_id별 대표 saves만
    playlist_saves = df.drop_duplicates(subset=["playlist_id"])[
        ["playlist_id", "saves"]
    ]

    # 3) 숫자 변환
    playlist_saves["saves_numeric"] = pd.to_numeric(
        playlist_saves["saves"], errors="coerce"
    )

    # 4) 빈칸/비수치 개수
    empty_count = playlist_saves["saves_numeric"].isna().sum()

    # 5) 유효 값만
    vals = playlist_saves["saves_numeric"].dropna().astype(float)

    # 6) 전체 통계(가로/정수)
    overall = summarize(vals)
    print("\n[Overall (playlist-level saves)]")
    print(f"{'empty_count':>12s}={int(empty_count):>8d}  " + nice_row(overall))

    # 7) 0~1000 구간 필터
    sub = vals[(vals >= 0) & (vals <= 1000)]
    sub_stats = summarize(sub) if len(sub) else pd.Series(dtype=int)
    print("\n[Subset 0~1000]")
    if len(sub):
        print(nice_row(sub_stats))
    else:
        print("no data in [0, 1000]")

    # 8) KDE + 피크
    if len(sub) >= 3:  # KDE에 최소 표본 안전장치
        x, y, peaks_idx, peak_points = kde_peaks_0_1000(sub)

        # 플롯
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label="KDE")
        if len(peaks_idx):
            plt.scatter(x[peaks_idx], y[peaks_idx], zorder=5, label="Peaks")
        plt.title("KDE of Saves (0 ~ 1000) with Peaks")
        plt.xlabel("Saves")
        plt.ylabel("Density")
        plt.xlim(0, 1000)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 상위 피크(밀도 큰 순) 정렬
        peak_points_sorted = sorted(peak_points, key=lambda t: t[1], reverse=True)
        print("\n[Peaks (0~1000), sorted by density desc]")
        for i, (px, py) in enumerate(peak_points_sorted, 1):
            print(f"{i:2d}) x≈{px:.2f}, density≈{py:.6f}")

        if peak_points_sorted:
            top_x, top_y = peak_points_sorted[0]
            print(f"\nTop peak: x≈{top_x:.2f}, density≈{top_y:.6f}")
    else:
        print("\nKDE 생략: 0~1000 구간 표본이 너무 적습니다.")


if __name__ == "__main__":
    main()

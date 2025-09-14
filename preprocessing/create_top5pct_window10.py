# create_top5pct_window10.py
# 목적: playlist_tracks_filtered.csv에서 윈도우 기반 이웃 정보 생성 → 상위 5% 필터링 → 윈도우 10 제한

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# tqdm (안 깔려 있으면 조용히 비활성)
try:
    from tqdm.auto import tqdm
    TQDM = True
except Exception:
    TQDM = False
    def tqdm(x=None, **_kwargs):
        return x if x is not None else iter([])


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


def most_common_or_first(series):
    """시리즈에서 가장 흔한 값을 반환, 동점이면 먼저 나온 것"""
    vals = [x for x in series if pd.notna(x) and str(x).strip() != ""]
    if not vals:
        return ""
    c = Counter(vals)
    return max(vals, key=lambda v: (c[v], -vals.index(v)))


def split_clean(s):
    """파이프로 구분된 문자열을 리스트로 분할"""
    if not isinstance(s, str) or not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def build_window_neighbors(df: pd.DataFrame, k: int):
    """각 playlist에서 track_index 순으로 정렬 후, 앞뒤 k곡을 이웃으로 수집"""
    if "track_index" not in df.columns:
        print("[경고] track_index 컬럼이 없어서 순서 기반 윈도우를 사용할 수 없습니다.")
        print("대신 플레이리스트 내 모든 곡을 이웃으로 처리합니다.")
        return build_playlist_neighbors(df)
    
    df = df.copy()
    df["track_index"] = pd.to_numeric(df["track_index"], errors="coerce")
    df = df.dropna(subset=["track_index"])
    df["track_index"] = df["track_index"].astype(int)

    neighbors = defaultdict(set)
    it = df.groupby("playlist_id", sort=False)
    if TQDM:
        it = tqdm(it, desc="윈도우 기반 이웃 구성", unit="playlist")
    
    for pid, g in it:
        g = g.sort_values("track_index")
        # 같은 곡 중복 제거(순서 보존)
        seen, ordered = set(), []
        for tid in g["track_id"].astype(str):
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
        
        L = len(ordered)
        if L <= 1:
            continue
            
        for i, tid in enumerate(ordered):
            lo = max(0, i - k)
            hi = min(L, i + k + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                neighbors[tid].add(ordered[j])
    
    return neighbors


def build_playlist_neighbors(df: pd.DataFrame):
    """플레이리스트 내 모든 곡을 이웃으로 처리 (track_index 없을 때)"""
    neighbors = defaultdict(set)
    it = df.groupby("playlist_id", sort=False)
    if TQDM:
        it = tqdm(it, desc="플레이리스트 기반 이웃 구성", unit="playlist")
    
    for pid, g in it:
        track_ids = list(set(g["track_id"].astype(str)))
        if len(track_ids) <= 1:
            continue
            
        for tid in track_ids:
            for other_tid in track_ids:
                if tid != other_tid:
                    neighbors[tid].add(other_tid)
    
    return neighbors


def aggregate_tracks(df: pd.DataFrame, window_k: int = 0) -> pd.DataFrame:
    """트랙별로 메타데이터와 플레이리스트 정보를 집계"""
    df = df.copy()
    need = {"playlist_id", "track_id"}
    if not need.issubset(df.columns):
        miss = need - set(df.columns)
        raise SystemExit(f"CSV에 필수컬럼 부족: {miss}")

    # 문자열 정리
    for col in ["playlist_id", "track_id", "track_title", "album_title", "artist_name", "artist_names"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .fillna("")
                .str.replace("\r", "", regex=False)
                .str.replace("\n", " ", regex=False)
            )
    
    df = df[(df["playlist_id"].str.strip() != "") & (df["track_id"].str.strip() != "")]

    # 아티스트 정규화
    if "artist_names" not in df.columns:
        df["artist_names"] = ""
    if "artist_name" not in df.columns:
        df["artist_name"] = ""
    
    a_multi = df["artist_names"].fillna("").astype(str)
    a_single = df["artist_name"].fillna("").astype(str)
    df["artists_norm"] = a_multi.where(a_multi.str.strip() != "", a_single)

    # 아티스트 리스트 파싱
    df["artists_list"] = df["artists_norm"].apply(split_clean)

    # 윈도우 이웃 구성
    neighbors = {}
    if window_k and window_k > 0:
        neighbors = build_window_neighbors(df, k=window_k)
    else:
        print("[정보] 윈도우 크기가 0이므로 이웃 정보를 생성하지 않습니다.")

    # 트랙별 집계
    n_groups = df["track_id"].nunique()
    it = df.groupby("track_id", sort=False)
    if TQDM:
        it = tqdm(it, total=n_groups, desc="트랙별 집계", unit="track")

    records = []
    for tid, g in it:
        pls = sorted(set(p for p in g["playlist_id"].astype(str) if p.strip()))
        count = len(pls)
        
        track = most_common_or_first(g.get("track_title", pd.Series(dtype=str)))
        album = most_common_or_first(g.get("album_title", pd.Series(dtype=str)))

        # 아티스트 합집합(순서 보존)
        artist_union = []
        for lst in g["artists_list"]:
            for a in lst:
                if a and a not in artist_union:
                    artist_union.append(a)
        if not artist_union:
            artist_union = [most_common_or_first(g.get("artist_name", pd.Series(dtype=str)))]

        nei_set = neighbors.get(tid, set())
        records.append({
            "track": track,
            "artist": "|".join(artist_union).strip("|"),
            "album": album,
            "track_id": tid,
            "playlist_id": "|".join(pls),
            "count": count,
            "neighbors": "|".join(nei_set) if nei_set else "",
            "pos_count_win": len(nei_set),
        })

    columns = ["track", "artist", "album", "track_id", "playlist_id", "count", "neighbors", "pos_count_win"]
    return pd.DataFrame.from_records(records, columns=columns)


def filter_top5pct_by_count(df: pd.DataFrame, percentile: float = 95.0) -> pd.DataFrame:
    """count 컬럼(플레이리스트 개수) 기준으로 상위 percentile% 이상인 트랙들만 필터링"""
    if "count" not in df.columns:
        raise ValueError("입력 데이터에 'count' 컬럼이 필요합니다.")
    
    df = df.copy()
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"])
    
    threshold = np.percentile(df["count"], percentile)
    print(f"플레이리스트 개수(count) 기준 {percentile}% 임계값: {threshold}")
    
    filtered = df[df["count"] >= threshold].copy()
    print(f"필터링 결과: {len(df):,} → {len(filtered):,} 곡 ({len(filtered)/len(df)*100:.1f}%)")
    
    return filtered


def apply_window_constraint(df: pd.DataFrame, window_k: int = 10) -> pd.DataFrame:
    """
    기존 neighbors 컬럼에서 윈도우 크기를 제한하여 새로운 컬럼들 생성
    필터링된 트랙들 중에서만 이웃으로 유지
    """
    if "neighbors" not in df.columns:
        print("[경고] neighbors 컬럼이 없습니다. 윈도우 제한을 건너뜁니다.")
        return df

    df = df.copy()
    
    # 필터링된 트랙 ID 집합 생성
    valid_track_ids = set(df["track_id"].astype(str))
    print(f"유효한 트랙 ID 수: {len(valid_track_ids):,}")

    print(f"윈도우 크기를 {window_k}로 제한하고 이웃 트랙 정보 검증 중...")

    new_neighbors = []
    new_pos_counts = []
    new_neighbor_info = []

    # 트랙 ID -> 메타데이터 매핑 생성
    track_meta = {}
    for _, row in df.iterrows():
        track_id = str(row["track_id"])
        track_meta[track_id] = {
            "track": row.get("track", ""),
            "artist": row.get("artist", ""),
            "album": row.get("album", "")
        }

    iterator = df["neighbors"].items()
    if TQDM:
        iterator = tqdm(iterator, total=len(df), desc="윈도우 제한 및 이웃 검증")

    for _idx, neighbors_str in iterator:
        if pd.isna(neighbors_str) or str(neighbors_str).strip() == "":
            new_neighbors.append("")
            new_pos_counts.append(0)
            new_neighbor_info.append("")
            continue

        neighbor_list = [x.strip() for x in str(neighbors_str).split("|") if x.strip()]

        # 유효한 이웃만 필터링 (필터링된 트랙들 중에서만)
        valid_neighbors = []
        neighbor_infos = []
        
        for neighbor_id in neighbor_list:
            if neighbor_id in valid_track_ids:
                valid_neighbors.append(neighbor_id)
                # 이웃 트랙 정보 추가
                meta = track_meta.get(neighbor_id, {"track": "", "artist": "", "album": ""})
                info = f"{neighbor_id}:{meta['track']}:{meta['artist']}"
                neighbor_infos.append(info)
                
                # 윈도우 크기 제한
                if len(valid_neighbors) >= window_k:
                    break

        new_neighbors.append("|".join(valid_neighbors))
        new_pos_counts.append(len(valid_neighbors))
        new_neighbor_info.append("||".join(neighbor_infos))

    df["neighbors_win10"] = new_neighbors
    df["pos_count_win10"] = new_pos_counts
    df["neighbor_info_win10"] = new_neighbor_info

    return df


def main():
    import argparse

    ap = argparse.ArgumentParser(description="트랙 집계 → 플레이리스트 개수 기준 상위 5% 필터링 → 윈도우 10 이웃 생성")
    ap.add_argument("--input", default="preprocessing/playlist_tracks_filtered.csv",
                    help="입력 CSV 파일 경로")
    ap.add_argument("--output", default="preprocessing/track_playlist_counts_top5pct_win10.csv",
                    help="최종 출력 CSV 파일 경로")
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="상위 몇 퍼센트를 유지할지 (default: 95.0 = 상위 5%)")
    ap.add_argument("--window-k", type=int, default=10,
                    help="윈도우 크기 (필터링 후 적용)")

    args = ap.parse_args()

    print("=== 1단계: 원본 데이터 로드 및 기본 집계 ===")
    print(f"[입력] {args.input}")
    df = robust_read_csv(args.input)
    print(f"원본 데이터: {len(df):,} 행")

    # 1단계: 기본 집계 (윈도우 없이)
    df_aggregated = aggregate_tracks(df, window_k=0)
    print(f"집계 완료: {len(df_aggregated):,} 곡")

    print(f"\n=== 2단계: 플레이리스트 개수 기준 상위 {100-args.percentile:.1f}% 필터링 ===")
    # 2단계: count 기준 상위 5% 필터링
    df_filtered = filter_top5pct_by_count(df_aggregated, percentile=args.percentile)

    print(f"\n=== 3단계: 필터링된 데이터에 윈도우 {args.window_k} 이웃 정보 생성 ===")
    # 3단계: 필터링된 트랙들만으로 윈도우 이웃 다시 생성
    filtered_track_ids = set(df_filtered["track_id"].astype(str))
    df_filtered_orig = df[df["track_id"].isin(filtered_track_ids)].copy()
    
    print(f"필터링된 트랙들의 원본 데이터: {len(df_filtered_orig):,} 행")
    df_with_window = aggregate_tracks(df_filtered_orig, window_k=args.window_k)
    
    # 필터링된 메타데이터와 윈도우 정보 합치기
    df_final = df_filtered.drop(columns=["neighbors", "pos_count_win"], errors="ignore")
    window_cols = df_with_window[["track_id", "neighbors", "pos_count_win"]]
    df_final = df_final.merge(window_cols, on="track_id", how="left")

    print(f"\n=== 4단계: 결과 저장 ===")
    # 4단계: 결과 저장
    print(f"[출력] {args.output}")
    df_final.to_csv(args.output, index=False)

    # 5단계: 통계 출력
    print(f"\n=== 최종 결과 요약 ===")
    print(f"원본: {len(df):,} 행")
    print(f"집계 후: {len(df_aggregated):,} 곡")
    print(f"플레이리스트 개수 기준 상위 {100-args.percentile:.1f}% 필터링 후: {len(df_filtered):,} 곡")
    print(f"최종: {len(df_final):,} 곡")

    if "pos_count_win" in df_final.columns:
        pos_counts = df_final["pos_count_win"].astype(float)
        avg_neighbors = pos_counts.mean()
        nonzero_neighbors = pos_counts[pos_counts > 0]
        print(f"\n이웃 정보:")
        print(f"- 평균 윈도우 이웃 수: {avg_neighbors:.2f}")
        print(f"- 이웃이 있는 곡 수: {len(nonzero_neighbors):,} ({len(nonzero_neighbors)/len(df_final)*100:.1f}%)")
        if len(nonzero_neighbors) > 0:
            print(f"- 이웃이 있는 곡들의 평균 이웃 수: {nonzero_neighbors.mean():.2f}")

    # save 컬럼 관련 코드 제거됨
        
    if "count" in df_final.columns:
        avg_count = df_final["count"].astype(float).mean()
        print(f"- 평균 플레이리스트 등장 횟수: {avg_count:.2f}")
        
    print(f"\n생성된 컬럼:")
    print(f"- neighbors: 윈도우 {args.window_k} 이웃 트랙 ID들")  
    print(f"- pos_count_win: 윈도우 이웃 수")


if __name__ == "__main__":
    main()
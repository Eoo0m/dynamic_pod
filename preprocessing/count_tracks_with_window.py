# count_tracks_with_window.py
# 목적: playlist_tracks_filtered.csv에서 트랙별 집계 + 윈도우 기반 이웃 정보 생성

import pandas as pd
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


def main():
    import argparse

    ap = argparse.ArgumentParser(description="트랙별 집계 및 윈도우 기반 이웃 정보 생성")
    ap.add_argument("--input", default="preprocessing/playlist_tracks_filtered.csv",
                    help="입력 CSV 파일 경로")
    ap.add_argument("--output", default="preprocessing/track_playlist_counts_with_window.csv",
                    help="출력 CSV 파일 경로")
    ap.add_argument("--window-k", type=int, default=10,
                    help="플레이리스트 내 앞뒤 k 곡만 양성 후보로 집계 (0이면 비활성)")

    args = ap.parse_args()

    print(f"[입력] {args.input}")
    df = robust_read_csv(args.input)
    print(f"원본 데이터: {len(df):,} 행")

    result = aggregate_tracks(df, window_k=args.window_k)
    
    print(f"[출력] {args.output}")
    result.to_csv(args.output, index=False)
    print(f"집계 완료: {len(result):,} 곡")
    
    if args.window_k > 0:
        has_neighbors = result[result["pos_count_win"] > 0]
        print(f"이웃이 있는 곡: {len(has_neighbors):,} ({len(has_neighbors)/len(result)*100:.1f}%)")
        if len(has_neighbors) > 0:
            avg_neighbors = has_neighbors["pos_count_win"].mean()
            print(f"평균 이웃 수: {avg_neighbors:.2f}")


if __name__ == "__main__":
    main()
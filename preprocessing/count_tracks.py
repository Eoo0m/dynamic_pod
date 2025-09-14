# aggregate_track_playlist_counts.py
import sys
import pandas as pd
from collections import Counter

# --- tqdm (optional) ---
try:
    from tqdm import tqdm, trange

    TQDM = True
except Exception:
    TQDM = False

    def tqdm(x=None, **k):
        return x

    def trange(*a, **k):  # fallback
        n = a[0] if a else 0
        return range(n)


def most_common_or_first(series):
    vals = [x for x in series if pd.notna(x) and str(x).strip() != ""]
    if not vals:
        return ""
    c = Counter(vals)
    # 빈도 우선, 같은 빈도면 먼저 나온 값
    return max(vals, key=lambda v: (c[v], -vals.index(v)))


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 필수 컬럼
    need = {"playlist_id", "track_id"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV에 필수컬럼 부족: {need - set(df.columns)}")

    # 문자열로 통일 + 공백/결측 제거
    for col in (
        "playlist_id",
        "track_id",
        "track_title",
        "album_title",
        "artist_name",
        "artist_names",
    ):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .fillna("")
                .str.replace("\r", "", regex=False)
                .str.replace("\n", " ", regex=False)
            )
    df = df[(df["playlist_id"].str.strip() != "") & (df["track_id"].str.strip() != "")]

    # 아티스트 정규화(멀티 지원, 벡터화)
    if "artist_names" not in df.columns:
        df["artist_names"] = ""
    if "artist_name" not in df.columns:
        df["artist_name"] = ""
    a_multi = df["artist_names"].fillna("").astype(str)
    a_single = df["artist_name"].fillna("").astype(str)
    df["artists_norm"] = a_multi.where(a_multi.str.strip() != "", a_single)

    # 리스트화
    def split_clean(s):
        if not isinstance(s, str) or not s:
            return []
        return [x.strip() for x in s.split("|") if x.strip()]

    if TQDM:
        from tqdm import tqdm as _tqdm

        _tqdm.pandas(desc="Split artists")
        df["artists_list"] = df["artists_norm"].progress_apply(split_clean)
    else:
        df["artists_list"] = df["artists_norm"].apply(split_clean)

    # 그룹 개수 파악(진행바 total용)
    n_groups = df["track_id"].nunique()

    recs = []
    it = df.groupby("track_id", sort=False)
    it = tqdm(it, total=n_groups, desc="Aggregating", unit="track") if TQDM else it

    for tid, g in it:
        # 등장 플레이리스트 집합
        pls = sorted(set(p for p in g["playlist_id"].astype(str) if p.strip()))
        count = len(pls)

        # 대표 메타
        track = most_common_or_first(g.get("track_title", pd.Series(dtype=str)))
        album = most_common_or_first(g.get("album_title", pd.Series(dtype=str)))

        # 아티스트 합집합(순서 보존)
        artist_union = []
        for lst in g["artists_list"]:
            for a in lst:
                if a and a not in artist_union:
                    artist_union.append(a)
        if not artist_union:
            # 멀티가 전혀 없으면 단일의 최빈값
            artist_union = [
                most_common_or_first(g.get("artist_name", pd.Series(dtype=str)))
            ]

        recs.append(
            {
                "track": track,
                "artist": "|".join(artist_union).strip("|"),
                "album": album,
                "track_id": tid,
                "playlist_id": "|".join(pls),
                "count": count,
            }
        )

    out = pd.DataFrame.from_records(
        recs, columns=["track", "artist", "album", "track_id", "playlist_id", "count"]
    )
    return out


# 견고 읽기 + 진행 출력
def robust_read_csv(path: str) -> pd.DataFrame:
    # 엔진 python + 불량 라인 스킵, 큰 파일에서도 안전
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


def main(in_csv: str, out_csv: str):
    if TQDM:
        print(f"[읽기] {in_csv} → pandas")
    df = robust_read_csv(in_csv)

    # 진행바 포함 집계
    out = aggregate(df)

    if TQDM:
        print(f"[쓰기] {out_csv} ({len(out):,} rows)")
    out.to_csv(out_csv, index=False)


if __name__ == "__main__":
    in_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "preprocessing/playlist_tracks_filtered.csv"
    )
    out_path = (
        sys.argv[2] if len(sys.argv) > 2 else "preprocessing/track_playlist_counts.csv"
    )
    main(in_path, out_path)

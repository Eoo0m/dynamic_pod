# keyword 분포 집계 (occurrence & 플레이리스트 기준)
import re
import pandas as pd
import matplotlib.pyplot as plt

# ===== 설정 =====
CSV_PATH = "crawler/spotify_playlists_unique.csv"  # 파일 경로
CASE_SENSITIVE = False  # 대소문자 구분 안 하려면 False
TOPN = 20  # 상위 N개 시각화

# ===== 로드 =====
df = pd.read_csv(CSV_PATH)

if "keyword" not in df.columns:
    raise SystemExit(
        "CSV에 'keyword' 컬럼이 필요합니다. (index, keyword, title, playlist_id, url)"
    )

# ===== 키워드 파싱 함수 =====
SEP = r"[|,;/]"  # 구분자: | , ; /


def split_keywords(s: str):
    if not isinstance(s, str):
        return []
    toks = [t.strip() for t in re.split(SEP, s) if t.strip()]
    return toks


# ===== explode (여러 키워드 → 행 늘리기) =====
tmp = df.copy()
tmp["__kw_list"] = tmp["keyword"].apply(split_keywords)
ex = tmp.explode("__kw_list", ignore_index=True)
ex = ex[ex["__kw_list"].notna() & (ex["__kw_list"] != "")]
ex = ex.rename(columns={"__kw_list": "kw"})

# 대소문자 정규화(옵션)
if not CASE_SENSITIVE:
    ex["kw"] = ex["kw"].str.lower()

# ===== 1) 발생 빈도 기반 분포 (explode 후 단순 횟수) =====
occ_counts = ex["kw"].value_counts()
occ_total = int(occ_counts.sum())
occ_df = occ_counts.rename("occurrences").to_frame()
occ_df["occ_%"] = (occ_df["occurrences"] / occ_total * 100).round(2)

# ===== 2) 플레이리스트 단위 분포 (중복 제거 후 '몇 개의 playlist에서 등장했나') =====
if "playlist_id" in ex.columns:
    pl_ex = ex.dropna(subset=["playlist_id"]).drop_duplicates(
        subset=["playlist_id", "kw"]
    )
    pl_counts = (
        pl_ex.groupby("kw")["playlist_id"].nunique().sort_values(ascending=False)
    )
    pl_total = df["playlist_id"].nunique()
    pl_df = pl_counts.rename("unique_playlists").to_frame()
    pl_df["pl_%"] = (pl_df["unique_playlists"] / pl_total * 100).round(2)
else:
    # playlist_id 없을 때 대비
    pl_df = pd.DataFrame(columns=["unique_playlists", "pl_%"])

# ===== 합치기 =====
stats = occ_df.join(pl_df, how="outer").fillna(0)
stats = stats.astype(
    {"occurrences": "int64", "unique_playlists": "int64"}, errors="ignore"
)
stats = stats.sort_values(by=["occurrences", "unique_playlists"], ascending=False)

# ===== 결과 미리보기 =====
print("=== 키워드 분포 (상위 100) ===")
print(stats.head(100))

# ===== CSV 저장(옵션) =====
stats.to_csv("keyword_distribution.csv", encoding="utf-8-sig")
print("\n[저장] keyword_distribution.csv")

# ===== 시각화(상위 TOPN) =====
# 1) 발생 빈도
stats.head(TOPN)["occurrences"].plot(kind="bar", rot=45)
plt.title(f"Top {TOPN} Keywords by Occurrence")
plt.tight_layout()
plt.show()

# 2) 플레이리스트 기준
if "unique_playlists" in stats.columns and stats["unique_playlists"].sum() > 0:
    stats.head(TOPN)["unique_playlists"].plot(kind="bar", rot=45)
    plt.title(f"Top {TOPN} Keywords by #Unique Playlists")
    plt.tight_layout()
    plt.show()

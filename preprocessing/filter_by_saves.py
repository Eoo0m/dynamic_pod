import pandas as pd

# CSV 불러오기
df = pd.read_csv("crawler/playlist_tracks.csv")

# saves를 숫자로 변환 (문자 → 숫자, 변환 안 되면 NaN)
df["saves_numeric"] = pd.to_numeric(df["saves"], errors="coerce")

# 조건: saves가 NaN이 아니고, 15 초과인 경우만 남김
filtered = df[(df["saves_numeric"].notna()) & (df["saves_numeric"] > 15)]

# 저장 (원래 컬럼 유지, 임시 컬럼 제거)
filtered = filtered.drop(columns=["saves_numeric"])
filtered.to_csv("preprocessing/playlist_tracks_filtered.csv", index=False)

print("✅ playlist_tracks_filtered.csv 저장 완료")

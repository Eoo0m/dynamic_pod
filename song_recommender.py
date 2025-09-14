import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SongRecommender:
    def __init__(self, meta_csv_path, embeddings_path, keys_path):
        self.meta_df = pd.read_csv(meta_csv_path)
        self.embeddings = np.load(embeddings_path)
        self.keys = np.load(keys_path)

        # track_id를 인덱스로 매핑하는 딕셔너리 생성
        self.track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self.keys)}

    def find_track_id_by_title(self, song_title):
        """노래 제목으로 track_id를 찾는 함수"""
        # 대소문자 무시하고 부분 일치 검색
        matches = self.meta_df[
            self.meta_df["track"].str.contains(song_title, case=False, na=False)
        ]

        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches.iloc[0]["track_id"]
        else:
            # 여러 개 일치하는 경우 정확한 일치를 찾거나 첫 번째 결과 반환
            exact_match = matches[matches["track"].str.lower() == song_title.lower()]
            if len(exact_match) > 0:
                return exact_match.iloc[0]["track_id"]
            else:
                print(f"여러 곡이 발견되었습니다:")
                for _, row in matches.iterrows():
                    print(f"- {row['track']} by {row['artist']}")
                return matches.iloc[0]["track_id"]

    def get_similar_songs(self, track_id, num_recommendations=20):
        """주어진 track_id와 유사한 노래들을 찾는 함수"""
        if track_id not in self.track_id_to_idx:
            return None

        # 해당 track의 임베딩 인덱스 찾기
        target_idx = self.track_id_to_idx[track_id]
        target_embedding = self.embeddings[target_idx].reshape(1, -1)

        # 모든 임베딩과의 코사인 유사도 계산
        similarities = cosine_similarity(target_embedding, self.embeddings)[0]

        # 자기 자신을 제외하고 가장 유사한 노래들의 인덱스 찾기
        similar_indices = np.argsort(similarities)[::-1][1 : num_recommendations + 1]

        # 추천 노래 정보 수집
        recommendations = []
        for idx in similar_indices:
            similar_track_id = self.keys[idx]
            track_info = self.meta_df[
                self.meta_df["track_id"] == similar_track_id
            ].iloc[0]
            recommendations.append(
                {
                    "track_id": similar_track_id,
                    "track": track_info["track"],
                    "artist": track_info["artist"],
                    "album": track_info["album"],
                    "similarity": similarities[idx],
                }
            )

        return recommendations

    def recommend_songs(self, song_title, num_recommendations=10):
        """노래 제목을 입력받아 추천 노래 리스트를 반환하는 메인 함수"""
        # 1. 노래 제목으로 track_id 찾기
        track_id = self.find_track_id_by_title(song_title)

        if track_id is None:
            return f"'{song_title}'와 일치하는 노래를 찾을 수 없습니다."

        # 원곡 정보 출력
        original_song = self.meta_df[self.meta_df["track_id"] == track_id].iloc[0]
        print(f"선택된 노래: {original_song['track']} by {original_song['artist']}")
        print(f"앨범: {original_song['album']}")
        print("-" * 50)

        # 2. 유사한 노래들 찾기
        recommendations = self.get_similar_songs(track_id, num_recommendations)

        if recommendations is None:
            return f"해당 노래의 임베딩을 찾을 수 없습니다."

        # 3. 추천 결과 출력
        print(f"'{original_song['track']}'와 유사한 노래 {num_recommendations}곡:")
        print()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['track']} - {rec['artist']}")
            print(f"    앨범: {rec['album']}")
            print(f"    유사도: {rec['similarity']:.4f}")
            print()

        return recommendations


def main():
    # 추천 시스템 초기화
    recommender = SongRecommender(
        meta_csv_path="/Users/eomjoonseo/dynamic_pod/contrastive_learning/contrastive_top5pct_win_meta.csv",
        embeddings_path="/Users/eomjoonseo/dynamic_pod/contrastive_learning/contrastive_top5pct_win_embeddings.npy",
        keys_path="/Users/eomjoonseo/dynamic_pod/contrastive_learning/contrastive_top5pct_win_keys.npy",
    )

    # 사용 예시
    song_title = input("추천받고 싶은 노래 제목을 입력하세요: ")
    recommender.recommend_songs(song_title)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class MusicRecommender:
    def __init__(self):
        self.music_database = self._load_music_from_csv()
        self.selected_songs = []
        
    def _load_music_from_csv(self) -> Dict[str, Dict]:
        try:
            df = pd.read_csv('/Users/eomjoonseo/dynamic_pod/contrastive_learning/contrastive_top5pct_win_meta.csv')
            embeddings = np.load('/Users/eomjoonseo/dynamic_pod/contrastive_learning/contrastive_top5pct_win_embeddings.npy')
            
            songs = {}
            
            for idx, row in df.iterrows():
                track_name = str(row['track']) if pd.notna(row['track']) else f"Unknown_{idx}"
                artist_name = str(row['artist']) if pd.notna(row['artist']) else "Unknown Artist"
                
                if idx < len(embeddings):
                    embedding_vector = embeddings[idx]
                    
                    songs[track_name] = {
                        "artist": artist_name,
                        "vector": embedding_vector,
                        "track_id": row['track_id'],
                        "album": str(row['album']) if pd.notna(row['album']) else "Unknown Album",
                        "pos_count": row['pos_count']
                    }
            
            print(f"로드된 곡 수: {len(songs)}")
            print(f"임베딩 차원: {embeddings.shape[1]}")
            return songs
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return {}
    
    def find_song(self, query: str) -> Tuple[str, Dict]:
        query_lower = query.lower()
        
        for song_name, song_data in self.music_database.items():
            if query_lower in song_name.lower() or song_name.lower() in query_lower:
                return song_name, song_data
        
        for song_name, song_data in self.music_database.items():
            if query_lower in song_data["artist"].lower() or song_data["artist"].lower() in query_lower:
                return song_name, song_data
        
        return None, None
    
    def get_similar_songs(self, target_song: str, count: int = 10) -> List[Tuple[str, Dict, float]]:
        if target_song not in self.music_database:
            return []
        
        target_vector = self.music_database[target_song]["vector"]
        similarities = []
        
        for song_name, song_data in self.music_database.items():
            if song_name != target_song:
                similarity = self._cosine_similarity(target_vector, song_data["vector"])
                similarities.append((song_name, song_data, similarity))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:count]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def add_selected_song(self, song_name: str):
        if song_name in self.music_database:
            self.selected_songs.append(song_name)
    
    def get_final_recommendations(self, count: int = 10) -> List[Tuple[str, Dict, float]]:
        if len(self.selected_songs) != 3:
            return []
        
        selected_vectors = [self.music_database[song]["vector"] for song in self.selected_songs]
        average_vector = np.mean(selected_vectors, axis=0)
        
        similarities = []
        for song_name, song_data in self.music_database.items():
            if song_name not in self.selected_songs:
                similarity = self._cosine_similarity(average_vector, song_data["vector"])
                similarities.append((song_name, song_data, similarity))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:count]

def main():
    recommender = MusicRecommender()
    
    print("🎵 음악 추천 시스템에 오신 것을 환영합니다!")
    print("좋아하는 곡을 하나 말씀해 주세요.\n")
    
    song_input = input("곡 제목을 입력하세요: ").strip()
    found_song, song_data = recommender.find_song(song_input)
    
    if not found_song:
        print(f"'{song_input}' 곡을 찾을 수 없습니다.")
        return
    
    print(f"\n'{found_song}' by {song_data['artist']} 을(를) 찾았습니다!")
    print("\n유사한 곡 10곡을 추천해 드립니다:")
    
    similar_songs = recommender.get_similar_songs(found_song, 10)
    
    for i, (song_name, song_info, similarity) in enumerate(similar_songs, 1):
        print(f"{i:2d}. {song_name} - {song_info['artist']} (유사도: {similarity:.3f})")
    
    while True:
        try:
            choice = input(f"\n마음에 드는 곡의 번호를 선택하세요 (1-10): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= 10:
                selected_song = similar_songs[choice_num - 1][0]
                recommender.add_selected_song(selected_song)
                print(f"'{selected_song}' 이(가) 선택되었습니다!\n")
                break
            else:
                print("1부터 10 사이의 번호를 입력해 주세요.")
        except ValueError:
            print("올바른 번호를 입력해 주세요.")
    
    for round_num in range(2, 4):
        print(f"=== {round_num}번째 곡 선택 ===")
        
        last_selected = recommender.selected_songs[-1]
        print(f"'{last_selected}' 기반으로 유사한 곡 10곡을 추천해 드립니다:")
        
        similar_songs = recommender.get_similar_songs(last_selected, 10)
        
        for i, (song_name, song_info, similarity) in enumerate(similar_songs, 1):
            print(f"{i:2d}. {song_name} - {song_info['artist']} (유사도: {similarity:.3f})")
        
        while True:
            try:
                choice = input(f"\n마음에 드는 곡의 번호를 선택하세요 (1-10): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= 10:
                    selected_song = similar_songs[choice_num - 1][0]
                    recommender.add_selected_song(selected_song)
                    print(f"'{selected_song}' 이(가) 선택되었습니다!\n")
                    break
                else:
                    print("1부터 10 사이의 번호를 입력해 주세요.")
            except ValueError:
                print("올바른 번호를 입력해 주세요.")
        
        print("-" * 50)
    
    print("\n=== 최종 추천 곡 ===")
    print(f"선택하신 3곡: {', '.join(recommender.selected_songs)}")
    print("\n이 3곡의 벡터 평균을 기반으로 추천하는 곡들입니다:")
    
    final_recommendations = recommender.get_final_recommendations(10)
    
    print("\n🎶 추천 곡 목록:")
    for i, (song_name, song_info, similarity) in enumerate(final_recommendations, 1):
        print(f"{i:2d}. {song_name} - {song_info['artist']} (매칭도: {similarity:.3f})")

if __name__ == "__main__":
    main()
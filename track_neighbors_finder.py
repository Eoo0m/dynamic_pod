import pandas as pd
import sys


def find_track_neighbors(track_name, csv_file, playlist_file):
    """
    곡 제목으로 neighbors를 찾고 neighbor들의 곡 이름과 플레이리스트 정보를 반환합니다.
    """
    df = pd.read_csv(csv_file)
    playlist_df = pd.read_csv(playlist_file)

    # 곡 제목으로 검색 (대소문자 구분 안함)
    matching_tracks = df[df["track"].str.contains(track_name, case=False, na=False)]

    if matching_tracks.empty:
        return f"'{track_name}' 곡을 찾을 수 없습니다."

    result = []
    for _, row in matching_tracks.iterrows():
        result.append(f"\n찾은 곡: {row['track']} - {row['artist']}")

        # 플레이리스트 정보 추가
        playlist_info = playlist_df[playlist_df["playlist_id"] == row["playlist_id"]]
        if not playlist_info.empty:
            playlist_name = playlist_info.iloc[0]["title"]
            result.append(f"플레이리스트: {playlist_name} (ID: {row['playlist_id']})")
        else:
            result.append(f"플레이리스트 ID: {row['playlist_id']}")

        # neighbors 파싱 (|로 구분된 track_id들)
        neighbors_str = str(row["neighbors"])
        if neighbors_str and neighbors_str != "nan":
            neighbor_ids = neighbors_str.split("|")

            # neighbor track_id들로 곡 이름 찾기
            neighbor_tracks = df[df["track_id"].isin(neighbor_ids)]

            result.append(f"유사한 곡들 ({len(neighbor_tracks)}개):")
            for _, neighbor in neighbor_tracks.iterrows():
                # 각 neighbor의 플레이리스트 정보도 가져오기
                neighbor_playlist = playlist_df[
                    playlist_df["playlist_id"] == neighbor["playlist_id"]
                ]
                if not neighbor_playlist.empty:
                    playlist_name = neighbor_playlist.iloc[0]["title"]
                    result.append(
                        f"  - {neighbor['track']} by {neighbor['artist']} [플레이리스트: {playlist_name}]"
                    )
                else:
                    result.append(
                        f"  - {neighbor['track']} by {neighbor['artist']} [플레이리스트 ID: {neighbor['playlist_id']}]"
                    )
        else:
            result.append("이 곡에 대한 neighbors 정보가 없습니다.")

    return "\n".join(result)


def main():
    csv_file = "/Users/eomjoonseo/dynamic_pod/preprocessing/track_playlist_counts_with_window.csv"
    playlist_file = "/Users/eomjoonseo/dynamic_pod/crawler/spotify_playlists_unique.csv"

    if len(sys.argv) > 1:
        track_name = " ".join(sys.argv[1:])
        print(find_track_neighbors(track_name, csv_file, playlist_file))
    else:
        print("사용법: python track_neighbors_finder.py <곡제목>")
        print("예시: python track_neighbors_finder.py WILDFLOWER")


if __name__ == "__main__":
    main()

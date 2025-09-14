# Dynamic Pod – Spotify Playlist & Track Embedding Pipeline

This repository contains the pipeline for crawling Spotify playlists/tracks, preprocessing, and analyzing track–playlist relationships.  
It enables large-scale playlist analysis and similarity benchmarking for music recommendation research.

---

## 📂 Data Schema

### Track Info
| column       | description              |
|--------------|--------------------------|
| track        | Track name               |
| artist       | Artist name              |
| album        | Album title              |
| track_id     | Spotify track ID         |
| playlist_id  | Parent playlist ID       |
| play_count   | Play count (if available)|

### Playlist Info
| column       | description              |
|--------------|--------------------------|
| index        | Playlist index           |
| keyword      | Search keyword           |
| title        | Playlist title           |
| playlist_id  | Spotify playlist ID      |
| url          | Spotify playlist URL     |
| saves        | Save count               |

---

## 🚀 Pipeline

### 1. Crawling
- **`playlist_scraper.py` → `playlists.csv`**  
  Stores playlist metadata (keyword, title, playlist_id, url).
- **`track_scraper.py` → `playlist_tracks.csv`**  
  Stores full track metadata (playlist info + track info + artist/album).

### 2. Preprocessing
- **`filter_by_saves.py`**  
  Keep only playlists with **saves ≥ 15**.  
- **`count_tracks.py` → `track_playlist_counts.csv`**  
  Count how many playlists each track appears in.  
- **`filter_top1.py` → `track_playlist_counts_preprocessed.csv`**  
  Cleaned track–playlist occurrence counts.

### 3. Statistics & Analysis
- **`analyze_playlist_saves.py`**  
  Analyze playlist saves distribution → detect/remove outliers.
  count=16495, mean=85,444, std=587,604, min=1, max=34,839,028

- **`show_pos_counts.py`**  
Distribution of neighbor counts (n=29,152, mean=213).  
- **`saves_distance_correlation.py`**  
Position–similarity correlation:  
- Adjacent tracks (distance=1) mean similarity = 0.5292  
- Distant tracks (distance≥15) mean similarity = 0.4909  
- p-value = 1.6e-71 (statistically significant).  
- **`benchmark_playlist_similarity.py`**  
Benchmark positive vs negative samples.  
- With windowing: margin mean = 0.3512  
- Without windowing: margin mean = 0.4265  
- T-test p < 1e-300 → strong separation.  

---

## 📊 Key Results
- Playlist saves show heavy-tailed distribution; threshold at **15 saves** improves data quality.  
- Neighboring tracks in playlists are significantly more similar than distant ones.  
- Benchmark confirms **>99.9% separation** between positive vs negative track pairs.  

---

## 🛠 Tech Stack
- Python 3.x  
- pandas, numpy, tqdm  
- Spotify Web API (via scrapers)  

---

## 🔮 Next Steps
- Train contrastive embedding model on filtered track–playlist pairs.  
- Evaluate recommendation quality with downstream dynamic playback.  

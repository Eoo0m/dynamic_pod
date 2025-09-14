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


### 1. Preprocessing
- **`filter_by_saves.py`**  
  Keep only playlists with **saves ≥ 15**.  
- **`count_tracks.py` → `track_playlist_counts.csv`**  
  Count how many playlists each track appears in.  
- **`filter_top1.py` → `track_playlist_counts_preprocessed.csv`**  
  Cleaned track–playlist occurrence counts.

### 2. Statistics & Analysis
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
-> 윈도우 없이 학습
---

## 📊 Key Results
- Playlist saves show heavy-tailed distribution; threshold at **15 saves** improves data quality.  
- Neighboring tracks in playlists are significantly more similar than distant ones.  
- Benchmark confirms **>99.9% separation** between positive vs negative track pairs.  

---

## 🎵 Contrastive Training

- **Method**: Contrastive learning with InfoNCE loss  
  - Positives: tracks that co-occur in playlists (neighbors)  
  - Negatives: all other tracks in the batch  
  - Model: embedding lookup table (L2-normalized)

### 🔧 Key Parameters
- `dim=256` → embedding dimension  
- `lr=1e-3` → AdamW learning rate  
- `hubcap=500` → cap to down-weight very frequent “hub” tracks that appear in many playlists, preventing them from dominating the learning process  


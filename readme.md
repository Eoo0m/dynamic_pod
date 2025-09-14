# Dynamic Pod – Spotify Playlist & Track Embedding Pipeline

This repository contains the pipeline for crawling Spotify playlists/tracks, preprocessing, and analyzing track–playlist relationships.  
It enables large-scale playlist analysis and similarity benchmarking for music recommendation research.

---

## 📂 Data Schema

### Track Info
| column      | description              |
|-------------|--------------------------|
| track       | Track name               |
| artist      | Artist name              |
| album       | Album title              |
| track_id    | Spotify track ID         |
| playlist_id | Parent playlist ID       |

### Playlist Info
| column      | description              |
|-------------|--------------------------|
| index       | Playlist index           |
| keyword     | Search keyword           |
| title       | Playlist title           |
| playlist_id | Spotify playlist ID      |
| url         | Spotify playlist URL     |
| saves       | Save count               |

---

## 🚀 Pipeline

### 1. Preprocessing
- **`filter_by_saves.py`**  
  Keep only playlists with **saves ≥ 15**.  
- **`count_tracks.py` → `track_playlist_counts.csv`**  
  Count how many playlists each track appears in.  
- **`filter_top1.py` → `track_playlist_counts_preprocessed.csv`**  
  Retain cleaned track–playlist occurrence counts.

### 2. Statistics & Analysis
- **`analyze_playlist_saves.py`**  
  - Total playlists analyzed: 16,495  
  - Empty entries: 4,724  
  - Mean saves: ~85K (skewed by a few highly popular playlists)  
  - Std dev: ~587K → strong long-tail distribution  
  - Median saves: 1,115 (much smaller than mean)  
  - Quartiles: Q25 = 87, Q75 = 10,209  
  - Range: 1 ~ 34,839,028  

  The density plot shows a first peak at **x ≈ 15.5**, indicating playlists with fewer than ~15 saves are likely low-quality or inactive.  
  ➡️ Threshold applied: **saves ≥ 15**.  

- **`benchmark_playlist_similarity.py`**  
### 🔹 With Windowing (윈도우 10)
- **Samples**: n = 29,151  
- **Positive pairs**: mean = 0.3467 (sd = 0.1106, median = 0.3267)  
- **Negative pairs**: mean = -0.0044 (sd = 0.0049, median = -0.0040)  
- **Margin (pos–neg)**: mean = 0.3512 (sd = 0.1095)  
- **Accuracy (margin > 0)**: 100%  
- **T-test**: t = 547.724, p ≈ 0  

**Overall track similarity** (5,000 sampled tracks, 12.5M pairs):  
- mean = 0.0001, std = 0.0887, median = -0.0110  
- min = -0.3178, max = 0.9090  


### 🔹 Without Windowing (윈도우 없이)
- **Samples**: n = 28,563  
- **Positive pairs**: mean = 0.4211 (sd = 0.1077, median = 0.4021)  
- **Negative pairs**: mean = -0.0054 (sd = 0.0058, median = -0.0049)  
- **Margin (pos–neg)**: mean = 0.4265 (sd = 0.1079)  
- **Accuracy (margin > 0)**: 99.99%  
- **T-test**: t = 668.029, p ≈ 0  

**Overall track similarity** (5,000 sampled tracks, 12.5M pairs):  
- mean = 0.0006, std = 0.0978, median = -0.0133  
- min = -0.3326, max = 0.9550  



## 📊 Preprocessing Based on Statistics
- Retain only playlists with **saves ≥ 15**.  
- From these playlists, keep only the **top 5% tracks by neighbor count**.  

---

## 3.Contrastive Training

- **Method**: Contrastive learning with InfoNCE loss  
  - Positives: tracks that co-occur in playlists (neighbors)  
  - Negatives: all other tracks in the batch  
  - Model: embedding lookup table (L2-normalized)  

### 🔧 Key Parameters
- `dim=256` → embedding dimension  
- `lr=1e-3` → AdamW learning rate  
- `hubcap=500` → cap to down-weight very frequent “hub” tracks that appear in many playlists, preventing them from dominating the training process  

---

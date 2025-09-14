
track info
track,artist,album,track_id,playlist_id,play_count

playlist_info
index,keyword,title,playlist_id,url,saves
0,chill,Chill Mix,37i9dQZF1EVHGWrwldPRtj,https://open.spotify.com/playlist/37i9dQZF1EVHGWrwldPRtj


# 1.crawling

## playlist_scraper.py -> playlists.csv
keyword,title,playlist_id,url
## track_scraper -> playlist_tracks.csv
playlist_index,playlist_id,playlist_title,playlist_url,saves,total_songs,track_index,track_title,track_id,track_url,artist_name,artist_id,artist_url,artist_names,artist_ids,artist_urls,album_title,album_id,album_url

# 2.preprocessing
## filter_by_saves -> save 15 이상인 플레이리스트만 남김.
## count_tracks.py -> track_playlist_counts.csv
Split artists: 100%|█████████████████████████████████████████████████████| 1146436/1146436 [00:02<00:00, 410760.10it/s]
Aggregating:   9%|█████▎                                                   | 53109/572430 [00:22<03:22, 2560.82track/s]
## filter_top1.py -> track_playlist_counts_preprocssed.csv



# 3.statistics
## analyze_playlist_saves -> 플레이리스트의 saves를 분석하여, 이상치(너무 낮은 값) 제거
 count  empty_count  mean    std   min  q25  median   q75      max
 16495         4724 85444  587604    1   87    1115 10209 34839028
 Peak 1: x ≈ 15.5, density ≈ 0.004226


## show_pos_counts

[요약] neighbors_count
n=29152  mean=213.0315  std=192.6291  median=144.0000  min=88  max=2566  p90=407.0000  p95=577.0000  p99=1073.4900

## saves_distance_correlation
=== PLAYLIST POSITION-SIMILARITY ANALYSIS ===

Total track pairs analyzed: 222,966
Number of playlists: 381

Correlation Analysis:
Pearson correlation: -0.0450 (p-value: 1.7422e-100)
Spearman correlation: -0.0473 (p-value: 8.5013e-111)

Adjacent vs Distant Tracks:
Adjacent tracks (distance=1) - Mean similarity: 0.5292 (±0.2187)
Distant tracks (distance>=15) - Mean similarity: 0.4909 (±0.2214)
T-test p-value: 1.6059e-71


Similarity Statistics by Position Distance (1-20):
================================================================================
Distance Count    Mean     Std      Var      Min      Max      Median  
--------------------------------------------------------------------------------
1        12958    0.5292   0.2187   0.0478   -0.2109  1.0000   0.5555  
2        12699    0.5169   0.2230   0.0497   -0.2528  1.0000   0.5422  
3        12462    0.5135   0.2220   0.0493   -0.2336  1.0000   0.5403  
4        12152    0.5132   0.2237   0.0500   -0.2295  1.0000   0.5387  
5        11955    0.5056   0.2228   0.0497   -0.2042  1.0000   0.5279  
6        11827    0.5069   0.2224   0.0495   -0.2265  1.0000   0.5326  
7        11555    0.4999   0.2205   0.0486   -0.2717  1.0000   0.5218  
8        11420    0.5029   0.2235   0.0500   -0.2100  1.0000   0.5214  
9        11341    0.4976   0.2209   0.0488   -0.2159  1.0000   0.5191  
10       11204    0.4967   0.2247   0.0505   -0.2069  1.0000   0.5172  
11       10952    0.4959   0.2231   0.0498   -0.1936  1.0000   0.5169  
12       10881    0.4927   0.2244   0.0503   -0.1994  1.0000   0.5128  
13       10656    0.4948   0.2225   0.0495   -0.2017  1.0000   0.5177  
14       10548    0.4940   0.2211   0.0489   -0.2138  1.0000   0.5132  
15       10383    0.4950   0.2214   0.0490   -0.1792  1.0000   0.5177  
16       10225    0.4922   0.2232   0.0498   -0.2098  1.0000   0.5142  
17       10112    0.4905   0.2205   0.0486   -0.2336  1.0000   0.5099  
18       10033    0.4888   0.2216   0.0491   -0.2360  1.0000   0.5073  
19       9818     0.4892   0.2200   0.0484   -0.2264  1.0000   0.5108  
20       9785     0.4895   0.2219   0.0492   -0.1612  1.0000   0.5095  
================================================================================



## benchmark_playlist_similarity
[샘플 수] n=29,151
[pos_mean] mean=0.3467 sd=0.1106 median=0.3267 q5=0.2043 q25=0.2670 q75=0.4058 q95=0.5611
[neg_mean] mean=-0.0044 sd=0.0049 median=-0.0040 q5=-0.0130 q25=-0.0072 q75=-0.0012 q95=0.0028
[margin]   mean=0.3512 sd=0.1095 median=0.3317 q5=0.2104 q25=0.2720 q75=0.4096 q95=0.5630
[정확도 유사 지표] margin>0 비율 = 100.00%
[t-test(pos vs neg)] t=547.724, p=0.00e+00

[전체 곡 유사도 통계] 샘플: 5,000곡, 쌍: 12,497,500개
[전체 유사도] mean=0.0001 std=0.0887 median=-0.0110
[전체 유사도] min=-0.3178 max=0.9090 q5=-0.1124 q25=-0.0533 q75=0.0354 q95=0.1508

[샘플 수] n=28,563
[pos_mean] mean=0.4211 sd=0.1077 median=0.4021 q5=0.2775 q25=0.3436 q75=0.4804 q95=0.6275
[neg_mean] mean=-0.0054 sd=0.0058 median=-0.0049 q5=-0.0153 q25=-0.0085 q75=-0.0016 q95=0.0033
[margin]   mean=0.4265 sd=0.1079 median=0.4088 q5=0.2807 q25=0.3489 q75=0.4868 q95=0.6328
[정확도 유사 지표] margin>0 비율 = 99.99%
[t-test(pos vs neg)] t=668.029, p=0.00e+00

[전체 곡 유사도 통계] 샘플: 5,000곡, 쌍: 12,497,500개
[전체 유사도] mean=0.0006 std=0.0978 median=-0.0133
[전체 유사도] min=-0.3326 max=0.9550 q5=-0.1137 q25=-0.0548 q75=0.0320 q95=0.1640




# 4.contrastive_learning




통계
평균 거리(모든 곡, 플리 내, 플리 외)
save - 평균거리 상관관계




개선점

추천받고 싶은 노래 제목을 입력하세요: LAST DANCE
선택된 노래: LAST DANCE by BIGBANG
앨범: MADE
--------------------------------------------------
'LAST DANCE'와 유사한 노래 10곡:

 1. Phonecert - 10CM
    앨범: 4.0
    유사도: 0.8278

 2. Y (Please Tell Me Why) - freestyle
    앨범: Freestyle 3
    유사도: 0.8183

 3. No Hope For Your Return - Jehwwn
    앨범: No Hope For Your Return
    유사도: 0.8164

 4. A Little Girl - OHHYUK
    앨범: Reply 1988 (Original Television Soundtrack), Pt. 3
    유사도: 0.8129

 5. To. X - TAEYEON
    앨범: To. X - The 5th Mini Album
    유사도: 0.8119

 6. Missing You - BTOB
    앨범: Brother Act.
    유사도: 0.8091

 7. What The Spring?? - 10CM
    앨범: What The Spring??
    유사도: 0.8000

 8. Stalker - 10CM
    앨범: 3.0
    유사도: 0.7923

 9. Slightly Tipsy (She is My Type♡ X SANDEUL) - Sandeul
    앨범: Slightly Tipsy (She is My Type♡ X SANDEUL)
    유사도: 0.7893

10. Cherry Blossom Ending - Busker Busker
    앨범: Busker Busker 1st
    유사도: 0.7845


-> 너무 시대,연배에 따라 추천하는 것이 아닌가?
연대 플리보다는 무드, 장르 더 추가해보기!

'BIRDS OF A FEATHER'와 유사한 노래 10곡:

 1. Die With A Smile - Lady Gaga|Bruno Mars
    앨범: Die With A Smile
    유사도: 0.9511

 2. That’s So True - Gracie Abrams
    앨범: The Secret of Us (Deluxe)
    유사도: 0.9265

 3. we can't be friends (wait for your love) - Ariana Grande
    앨범: eternal sunshine
    유사도: 0.9055

 4. Ordinary - Alex Warren
    앨범: Ordinary
    유사도: 0.9015

 5. End of Beginning - Djo
    앨범: DECIDE
    유사도: 0.8960

 6. Beautiful Things - Benson Boone
    앨범: Beautiful Things
    유사도: 0.8887

 7. Cruel Summer - Taylor Swift
    앨범: Lover
    유사도: 0.8846

 8. Please Please Please - Sabrina Carpenter
    앨범: Please Please Please
    유사도: 0.8823

 9. Good Luck, Babe! - Chappell Roan
    앨범: Good Luck, Babe!
    유사도: 0.8803

10. As It Was - Harry Styles
    앨범: Harry's House
    유사도: 0.8771

'Make You Feel My Love'와 유사한 노래 10곡:

 1. Little Things - One Direction
    앨범: Take Me Home (Expanded Edition)
    유사도: 0.6213

 2. Put Your Records On - Corinne Bailey Rae
    앨범: Corinne Bailey Rae
    유사도: 0.6091

 3. Sparks - Coldplay
    앨범: Parachutes
    유사도: 0.6062

 4. a thousand years - Christina Perri
    앨범: a thousand years
    유사도: 0.6013

 5. What a Wonderful World - Kina Grannis|Imaginary Future
    앨범: What a Wonderful World
    유사도: 0.5948

 6. Kiss Me - Sixpence None The Richer
    앨범: Sixpence None The Richer
    유사도: 0.5918

 7. Here With Me - d4vd
    앨범: Here With Me
    유사도: 0.5905

 8. Your Body Is a Wonderland - John Mayer
    앨범: Room For Squares
    유사도: 0.5903

 9. Make You Feel My Love - Adele
    앨범: 19
    유사도: 0.5892

10. Marry Me - Train
    앨범: Save Me, San Francisco (Golden Gate Edition)
    유사도: 0.5861

-> 인기곡은 유사도가 많음..!
인기곡, 플리가 지배하는 현상 조심

'Here Comes The Sun - Remastered 2009'와 유사한 노래 10곡:

 1. Brown Eyed Girl - Van Morrison
    앨범: Blowin' Your Mind!
    유사도: 0.9312

 2. Happy Together - The Turtles
    앨범: Happy Together
    유사도: 0.8838

 3. Surfin' U.S.A. - Remastered 2001 - The Beach Boys
    앨범: Surfin' USA (Remastered)
    유사도: 0.8709

 4. Mrs. Robinson - From "The Graduate" Soundtrack - Simon & Garfunkel
    앨범: Bookends
    유사도: 0.8606

 5. Wouldn't It Be Nice - The Beach Boys
    앨범: Pet Sounds (Original Mono & Stereo Mix)
    유사도: 0.8599

 6. Hotel California - 2013 Remaster - Eagles
    앨범: Hotel California (2013 Remaster)
    유사도: 0.8447

 7. Good Vibrations - Remastered 2001 - The Beach Boys
    앨범: Smiley Smile (Remastered)
    유사도: 0.8436

 8. Sweet Home Alabama - Lynyrd Skynyrd
    앨범: Second Helping (Expanded Edition)
    유사도: 0.8430

 9. Rocket Man (I Think It's Going To Be A Long, Long Time) - Elton John
    앨범: Honky Chateau
    유사도: 0.8288

10. Twist And Shout - Remastered 2009 - The Beatles
    앨범: Please Please Me (Remastered)
    유사도: 0.8231
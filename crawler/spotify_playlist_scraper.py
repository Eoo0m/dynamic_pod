# spotify_playlist_scraper.py
import asyncio, csv, re, urllib.parse, random
from pathlib import Path
from collections import OrderedDict
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

BASE = "https://open.spotify.com"
PLAYLIST_A = 'a[href^="/playlist/"]'  # 플레이리스트 카드 앵커
TITLE_IN_CARD = "p[title]"  # 카드 내 제목 요소


def build_url(keyword: str) -> str:
    return f"{BASE}/search/{urllib.parse.quote(keyword)}/playlists"


async def scrape_keyword(
    page, keyword: str, max_steps: int = 60, target_count: int = 300
):
    await page.goto(build_url(keyword), wait_until="domcontentloaded")
    await page.wait_for_selector(PLAYLIST_A, timeout=15000)

    stagnant = 0
    for i in range(max_steps):
        loc = page.locator(PLAYLIST_A)
        count_before = await loc.count()
        if count_before == 0:
            break

        # 목표 개수 도달 시 즉시 중단
        if count_before >= target_count:
            print(f"[{keyword}] {count_before}개 도달 → 중단")
            break

        try:
            await loc.nth(count_before - 1).scroll_into_view_if_needed(timeout=5000)
        except PlaywrightTimeoutError:
            pass

        await page.wait_for_timeout(random.uniform(800, 1500))
        count_after = await loc.count()

        if count_after == count_before:
            stagnant += 1
        else:
            stagnant = 0

        if stagnant >= 3:
            print(f"[{keyword}] 스크롤 {i+1}회 → {count_after}개에서 멈춤")
            break

    final_cnt = await page.locator(PLAYLIST_A).count()
    print(f"[{keyword}] 최종 플레이리스트 개수: {final_cnt}")

    # === 추출 ===
    rows = OrderedDict()  # playlist_id 기준 중복 제거
    anchors = page.locator(PLAYLIST_A)
    n = min(final_cnt, target_count)  # 300개로 제한
    for i in range(n):
        a = anchors.nth(i)
        href = await a.get_attribute("href")
        if not href:
            continue
        m = re.search(r"/playlist/([A-Za-z0-9]+)", href)
        if not m:
            continue
        pid = m.group(1)

        title = ""
        t_el = await a.locator(TITLE_IN_CARD).first.element_handle()
        if t_el:
            title = await t_el.get_attribute("title") or ""
        if not title:
            text = (await a.text_content() or "").strip()
            if not text:
                continue
            title = text.splitlines()[0].strip()

        rows[pid] = {
            "keyword": keyword,
            "title": title,
            "playlist_id": pid,
            "url": href if href.startswith("http") else f"{BASE}{href}",
        }

    return list(rows.values())


async def run(
    keywords, out_csv="spotify_playlists.csv", headless=True, target_count: int = 300
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=["--no-sandbox"])
        ctx = await browser.new_context(viewport={"width": 1440, "height": 1000})
        page = await ctx.new_page()
        all_rows = []

        for kw in keywords:
            print(f"[+] {kw}")
            rows = await scrape_keyword(page, kw, target_count=target_count)
            if rows:
                all_rows.extend(rows)

        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["keyword", "title", "playlist_id", "url"])
            w.writeheader()
            w.writerows(all_rows)

        await browser.close()
        print(f"Saved {len(all_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    KEYWORDS = [
        # === 분위기 / 무드 ===
        "chill",
        "lofi",
        "study music",
        "focus",
        "sleep",
        "relax",
        "calm",
        "sad songs",
        "happy vibes",
        "energetic",
        "romantic songs",
        "love songs",
        "cozy",
        "motivational",
        "meditation",
        "mindfulness",
        "nostalgia",
        "rainy day",
        "morning coffee",
        "late night drive",
        "deep focus",
        "background music",
        "instrumental vibes",
        "dark academia",
        "aesthetic",
        # === 활동 / 상황 ===
        "workout",
        "gym",
        "running",
        "yoga",
        "pilates",
        "cycling",
        "hiking",
        "party",
        "club",
        "festival",
        "road trip",
        "driving",
        "car music",
        "dinner music",
        "cooking music",
        "office music",
        "study beats",
        "gaming music",
        "streaming background",
        "sleep sounds",
        "baby sleep",
        "kids songs",
        "family time",
        "wedding",
        "date night",
        "barbecue",
        "summer beach",
        "winter vibes",
        "christmas party",
        "new year party",
        # === 장르 ===
        "jazz",
        "smooth jazz",
        "classical",
        "orchestral",
        "opera",
        "pop hits",
        "k-pop",
        "hip hop",
        "r&b",
        "rock",
        "indie",
        "indie pop",
        "indie rock",
        "edm",
        "house",
        "deep house",
        "techno",
        "trance",
        "drum and bass",
        "dubstep",
        "reggaeton",
        "latin hits",
        "afrobeat",
        "dancehall",
        "city pop",
        "j-pop",
        "anime songs",
        "k-indie",
        "acoustic",
        "folk",
        "country",
        "blues",
        "soul",
        "funk",
        "gospel",
        "christian music",
        "metal",
        "punk rock",
        "emo",
        # === 시대 / 스타일 ===
        "latest release",
        "today’s top hits",
        "new release",
        "trending now",
        "viral hits",
        "tiktok songs",
        "chart toppers",
        "global top 50",
        "billboard hot 100",
        "hot trending songs",
        # === 레트로 / 회상 ===
        "retro",
        "throwback",
        "oldies",
        "classic rock",
        "golden oldies",
        "disco fever",
        "funk classics",
        "motown hits",
        "y2k pop",
        "indie sleaze",
        "nostalgic songs",
        "retro workout",
        "retro party",
        "retro vibes",
        # 60s
        "60s hits",
        "60s rock",
        "60s soul",
        "60s motown",
        # 70s
        "70s hits",
        "70s disco",
        "70s funk",
        "70s classic rock",
        "70s soul",
        # 80s
        "80s hits",
        "80s synthpop",
        "80s rock",
        "80s new wave",
        "80s metal",
        "80s ballads",
        # 90s
        "90s hits",
        "90s pop",
        "90s hip hop",
        "90s r&b",
        "90s grunge",
        "90s dance",
        # 2000s
        "2000s hits",
        "2000s pop",
        "2000s hip hop",
        "2000s r&b",
        "2000s emo",
        "2000s indie rock",
        "2000s k-pop",
        # 2010s
        "2010s hits",
        "2010s pop",
        "2010s edm",
        "2010s hip hop",
        "2010s k-pop",
        "2010s alternative",
        # 2020s
        "2020s hits",
        "2020s pop",
        "2020s hip hop",
        "2020s k-pop",
        "2020s viral",
        "2020s edm",
    ]

    # 키워드당 최대 300개로 제한
    asyncio.run(run(KEYWORDS, headless=True, target_count=300))

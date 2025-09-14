# scrape_playlist_tracks_onecsv.py
import os, asyncio, re, random, urllib.parse, argparse
from pathlib import Path
import pandas as pd
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

INPUT_CSV = "crawler/spotify_playlists_unique.csv"  # must contain column: url
OUT_CSV = "crawler/playlist_tracks.csv"  # single CSV (duplicates allowed)
CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))

BASE = "https://open.spotify.com"
SEL_TRACK_A = 'a[href^="/track/"]'
SEL_ARTIST_A = 'a[href^="/artist/"]'
SEL_ALBUM_A = 'a[href^="/album/"]'


# ---------- 공통 ----------
def abs_url(u: str) -> str:
    return (
        u
        if isinstance(u, str) and u.startswith("http")
        else urllib.parse.urljoin(BASE, u or "")
    )


def pick_id(href: str, kind: str) -> str:
    m = re.search(rf"/{kind}/([A-Za-z0-9]+)", href or "")
    return m.group(1) if m else ""


# ---------- 저장수(좋아요 수) ----------
def _to_int_num(s: str):
    s = (s or "").strip().replace(",", "").replace(" ", "")
    m = re.match(r"^(\d+(?:\.\d+)?)([kKmM]?)$", s)
    if not m:
        return None
    v, suf = m.groups()
    v = float(v)
    if suf.lower() == "k":
        v *= 1_000
    elif suf.lower() == "m":
        v *= 1_000_000
    return int(v)


async def get_save_count(page):
    texts = await page.eval_on_selector_all(
        "span[data-encore-id='text'], h1, h2, h3, div, button",
        "els => els.map(e => (e.textContent || '').trim())",
    )
    for t in texts:
        m = re.search(r"(\d[\d\.\s,]*[kKmM]?)\s*(?:saves?|likes?)\b", t, re.I)
        if m:
            n = _to_int_num(m.group(1))
            if n is not None:
                return n
    for t in texts:
        m = re.search(r"(?:좋아요)\s*([\d\.,]+)", t)
        if m:
            n = _to_int_num(m.group(1))
            if n is not None:
                return n
        m = re.search(r"([\d\.,]+)\s*명이\s*좋아합니다", t)
        if m:
            n = _to_int_num(m.group(1))
            if n is not None:
                return n
    # 헤더 근처 재스캔
    texts2 = await page.eval_on_selector_all(
        "[data-testid='entity-header'], [data-testid='playlist-page'], header",
        "els => els.flatMap(el => Array.from(el.querySelectorAll('*')).map(n => (n.textContent || '').trim()))",
    )
    for t in texts2:
        m = re.search(r"(\d[\d\.\s,]*[kKmM]?)\s*(?:saves?|likes?)\b", t, re.I)
        if m:
            n = _to_int_num(m.group(1))
            if n is not None:
                return n
        m = re.search(r"(?:좋아요)\s*([\d\.,]+)", t)
        if m:
            n = _to_int_num(m.group(1))
            if n is not None:
                return n
        m = re.search(r"([\d\.,]+)\s*명이\s*좋아합니다", t)
        if m:
            n = _to_int_num(m.group(1))
            if n is not None:
                return n
    return None


async def get_track_playcount(page, track_url):
    try:
        await page.goto(track_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)

        texts = await page.eval_on_selector_all(
            "span[data-encore-id='text'], div[data-testid='track-page'], [data-testid='entity-header']",
            "els => els.map(e => (e.textContent || '').trim())",
        )

        for t in texts:
            m = re.search(r"(\d[\d\.\s,]*[kKmM]?)\s*(?:plays?|streams?)\b", t, re.I)
            if m:
                n = _to_int_num(m.group(1))
                if n is not None:
                    return n
        return None
    except Exception:
        return None


# ---------- 곡 수(메타 텍스트) ----------
_PAT_TRACKS = [re.compile(r"([\d\.,]+)\s*곡"), re.compile(r"([\d\.,]+)\s*songs?", re.I)]


async def get_track_total_meta(page):
    loc = page.locator("//span[@data-encore-id='text']")
    cnt = await loc.count()
    texts = [(await loc.nth(i).inner_text() or "") for i in range(min(cnt, 200))]
    for t in texts:
        for p in _PAT_TRACKS:
            m = p.search(t)
            if m:
                return _to_int_num(m.group(1))
    return None


# ---------- grid에서 총 곡수 ----------
async def get_total_from_grid(page):
    grid = page.locator("div[role='grid']").first
    if await grid.count():
        v = await grid.get_attribute("aria-rowcount")
        if v and v.isdigit():
            iv = int(v)
            return iv if iv >= 1 else None
    return None


# ---------- 페이지 내에서 스크롤+수집(왕복 최소화) ----------
SCROLL_AND_COLLECT_JS = """
async ({goal, maxRounds, stagnantLimit, pauseMin, pauseMax}) => {
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const abs = (u) => u && u.startsWith('http') ? u : (new URL(u, 'https://open.spotify.com')+'');
  const cont = (() => {
    const els = Array.from(document.querySelectorAll('div'));
    let best = document.scrollingElement || document.body, maxH = 0;
    for (const el of els) {
      const st = getComputedStyle(el);
      if (/(auto|scroll)/.test(st.overflowY) && el.scrollHeight > el.clientHeight) {
        if (el.scrollHeight > maxH) { maxH = el.scrollHeight; best = el; }
      }
    }
    return best;
  })();

  const seen = new Set();
  const rows = [];
  let stagnant = 0;

  const snapshot = () => {
    const out = [];
    const anchors = Array.from(document.querySelectorAll('a[href^="/track/"]'));
    for (const a of anchors) {
      const trackHref = a.getAttribute('href') || '';
      if (!trackHref) continue;
      const trackTitle = (a.textContent || '').trim();

      const container = a.closest('[role="row"]')
        || a.closest('[data-testid="tracklist-row"]')
        || a.closest('[data-testid="tracklist-row-container"]')
        || a.closest('div');

      let rowIdx = null;
      if (container) {
        const idx = container.getAttribute('aria-rowindex');
        if (idx && /^\\d+$/.test(idx)) rowIdx = parseInt(idx, 10);
      }

      const arts = container ? Array.from(container.querySelectorAll('a[href^="/artist/"]')) : [];
      const aSeen = new Set();
      const artistNames = [], artistIds = [], artistUrls = [];
      for (const ai of arts) {
        const href = ai.getAttribute('href') || '';
        const m = href.match(/\\/artist\\/([A-Za-z0-9]+)/);
        const id = m ? m[1] : '';
        if (!id || aSeen.has(id)) continue;
        aSeen.add(id);
        artistNames.push((ai.textContent || '').trim());
        artistIds.push(id);
        artistUrls.push(href);
      }

      const alb = container ? container.querySelector('a[href^="/album/"]') : null;
      let albumTitle = '', albumId = '', albumUrl = '';
      if (alb) {
        const href = alb.getAttribute('href') || '';
        const m = href.match(/\\/album\\/([A-Za-z0-9]+)/);
        albumId = m ? m[1] : '';
        albumTitle = (alb.textContent || '').trim();
        albumUrl = href;
      }

      const key = rowIdx != null ? `idx:${rowIdx}` : `${trackHref}|${(artistIds[0]||'')}|${albumId}`;
      if (seen.has(key)) continue;
      seen.add(key);

      out.push({
        rowIdx,
        trackHref: abs(trackHref),
        trackTitle,
        artistNames,
        artistIds,
        artistUrls: artistUrls.map(abs),
        albumTitle, albumId, albumUrl: abs(albumUrl),
      });
    }
    return out;
  };

  for (let round = 0; round < maxRounds; round++) {
    const snap = snapshot();
    if (snap.length) {
      rows.push(...snap);
      stagnant = 0;
    } else {
      stagnant++;
    }

    if (goal && rows.length >= goal) break;

    if (stagnant >= stagnantLimit) {
      cont.scrollTop = cont.scrollHeight;            // 끝으로 점프
      await sleep(400);
      const after = snapshot();
      if (!after.length) break;                      // 더 못불러오면 종료
      rows.push(...after);
      stagnant = 0;
      if (goal && rows.length >= goal) break;
    }

    const step = cont.clientHeight * 0.95;
    cont.scrollBy(0, step);
    await sleep(Math.floor(pauseMin + Math.random() * (pauseMax - pauseMin)));
  }

  // 인덱스 기준 정렬/보정
  rows.sort((a,b) => {
    const ia = a.rowIdx == null ? 9e9 : a.rowIdx;
    const ib = b.rowIdx == null ? 9e9 : b.rowIdx;
    return ia - ib;
  });
  let nextIdx = 1;
  for (const r of rows) if (r.rowIdx == null) r.rowIdx = nextIdx++;
  return rows;
}
"""


# ---------- 단일 URL 처리 ----------
async def scrape_one(page, url: str, playlist_index: int):
    url = abs_url(url)
    await page.goto(url, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector(SEL_TRACK_A, timeout=15000)
    except PlaywrightTimeoutError:
        try:
            await page.wait_for_load_state("networkidle", timeout=4000)
            await page.wait_for_selector(SEL_TRACK_A, timeout=4000)
        except PlaywrightTimeoutError:
            return []

    playlist_id = pick_id(url, "playlist")
    title = (
        (await page.locator("h1").first.text_content() or "").strip()
        if await page.locator("h1").count()
        else ""
    )
    saves = await get_save_count(page)

    total_meta = await get_total_from_grid(page)
    if total_meta is None:
        total_meta = await get_track_total_meta(page)
    if total_meta == 0:
        total_meta = None

    # 200곡 초과면 스킵(정책)
    if total_meta is not None and total_meta > 200:
        print(
            f"[skip] {title} ({playlist_id}) | {total_meta}곡 → 200곡 이상, 스킵 | 저장수 {saves if saves is not None else 'N/A'}"
        )
        return []

    # 페이지 내부에서 한 번에 스크롤+수집
    snap = await page.evaluate(
        SCROLL_AND_COLLECT_JS,
        dict(
            goal=total_meta or 0,
            maxRounds=600,
            stagnantLimit=2,
            pauseMin=120,
            pauseMax=260,
        ),
    )

    # Python 측에서 가공
    rows = []
    for r in snap:
        track_id = pick_id(r.get("trackHref", ""), "track")
        artist_names = r.get("artistNames") or []
        artist_ids = r.get("artistIds") or []
        artist_urls = r.get("artistUrls") or []
        rows.append(
            {
                "playlist_index": playlist_index,
                "playlist_id": playlist_id,
                "playlist_title": title,
                "playlist_url": url,
                "saves": saves,
                "total_songs": total_meta,
                "track_index": r.get("rowIdx"),
                "track_title": r.get("trackTitle") or "",
                "track_id": track_id,
                "track_url": r.get("trackHref") or "",
                # 단일 + 멀티
                "artist_name": artist_names[0] if artist_names else "",
                "artist_id": artist_ids[0] if artist_ids else "",
                "artist_url": artist_urls[0] if artist_urls else "",
                "artist_names": "|".join(artist_names),
                "artist_ids": "|".join(artist_ids),
                "artist_urls": "|".join(artist_urls),
                "album_title": r.get("albumTitle") or "",
                "album_id": r.get("albumId") or "",
                "album_url": r.get("albumUrl") or "",
            }
        )
    return rows


# ---------- 개별(병렬용) ----------
async def scrape_with_new_page(ctx, url: str, playlist_index: int):
    page = await ctx.new_page()
    page.set_default_timeout(10000)
    try:
        await page.add_style_tag(
            content="*,*::before,*::after{transition:none!important;animation:none!important}"
        )
    except Exception:
        pass
    try:
        rows = await scrape_one(page, url, playlist_index)
    finally:
        await page.close()
    return url, rows, playlist_index


# ---------- 메인 ----------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="시작 인덱스 (지정하지 않으면 자동으로 이어서 시작)",
    )
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)
    if "url" not in df.columns:
        raise SystemExit("INPUT CSV에 'url' 컬럼이 필요합니다.")

    header_exists = Path(OUT_CSV).exists()

    # 시작 인덱스 결정
    if args.start is None:
        if header_exists:
            try:
                existing_df = pd.read_csv(OUT_CSV)
                if "playlist_index" in existing_df.columns and len(existing_df) > 0:
                    last_index = existing_df["playlist_index"].max()
                    args.start = last_index + 1
                    print(
                        f"기존 파일에서 마지막 인덱스 {last_index} 발견, 인덱스 {args.start}부터 이어서 시작"
                    )
                else:
                    args.start = 0
                    print(
                        "기존 파일에 playlist_index 컬럼이 없거나 비어있음, 인덱스 0부터 시작"
                    )
            except Exception as e:
                args.start = 0
                print(f"기존 파일 읽기 실패 ({e}), 인덱스 0부터 시작")
        else:
            args.start = 0
            print("새 파일 생성, 인덱스 0부터 시작")
    else:
        print(f"수동 지정된 인덱스 {args.start}부터 시작")

    if args.start > 0:
        df = df[df["index"] >= args.start]
        print(f"처리할 플레이리스트: {len(df)}개")

    urls_with_index = [
        (row["url"], row["index"]) for _, row in df.iterrows() if pd.notna(row["url"])
    ]

    launch_args = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=launch_args)
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 1800},
            reduced_motion="reduce",
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        )

        # 리소스 차단(이미지/미디어/폰트)
        BLOCK = {"image", "media", "font"}

        async def _route(route):
            try:
                if route.request.resource_type in BLOCK:
                    await route.abort()
                else:
                    await route.continue_()
            except Exception:
                try:
                    await route.continue_()
                except Exception:
                    pass

        await ctx.route("**/*", _route)

        sem = asyncio.Semaphore(max(1, min(CONCURRENCY, len(urls_with_index))))

        async def bounded(url_idx):
            url, idx = url_idx
            async with sem:
                return await scrape_with_new_page(ctx, url, idx)

        tasks = [asyncio.create_task(bounded(url_idx)) for url_idx in urls_with_index]
        for fut in asyncio.as_completed(tasks):
            url, rows, playlist_index = await fut
            if not rows:
                print(f"[!] 건너뜀 [인덱스 {playlist_index}]: {url}")
                continue

            pid = rows[0]["playlist_id"]
            title = rows[0]["playlist_title"]
            saves = rows[0]["saves"]
            total = rows[0]["total_songs"]
            found = len(rows)
            total_disp = (
                total if (isinstance(total, int) and total > 0) else f"~{found}"
            )
            print(
                f"[+] [인덱스 {playlist_index}] {title} ({pid}) | 전체 {total_disp}곡 | 추가 {found}곡 | 저장수 {saves if saves is not None else 'N/A'}"
            )

            pd.DataFrame(rows).to_csv(
                OUT_CSV, mode="a", index=False, header=not header_exists
            )
            header_exists = True

        await browser.close()
    print(f"[✓] 완료: {OUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())

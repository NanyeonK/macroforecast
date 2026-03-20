"""Scrape and cache Coulombe's blog posts via the Squarespace JSON API.

The endpoint returns a JSON object whose `items` list contains blog post
records.  Each record provides `title`, `body` (HTML), `publishOn`
(Unix ms timestamp), and `urlId`.

Run directly:
    uv run python macrocast/mcp/ingest_blog.py
"""

import json
import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from macrocast.mcp.config import BLOG_API_URL, BLOG_CACHE_DIR

logger = logging.getLogger(__name__)

BLOG_INDEX_FILE = BLOG_CACHE_DIR / "index.json"


def html_to_text(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse blank lines
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n\n".join(non_empty)


def fetch_posts(url: str = BLOG_API_URL) -> list[dict]:
    """Fetch all blog posts from the Squarespace JSON API.

    Returns a list of cleaned post dicts with keys:
    ``title``, ``url_id``, ``published_date``, ``text``.
    """
    logger.info("Fetching blog posts from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    items = data.get("items", [])
    logger.info("Found %d posts in API response.", len(items))

    posts = []
    for item in items:
        raw_html = item.get("body", "")
        text = html_to_text(raw_html)
        # publishOn is milliseconds since epoch
        published_ms = item.get("publishOn", 0)
        published_date = (
            time.strftime("%Y-%m-%d", time.gmtime(published_ms / 1000))
            if published_ms
            else "unknown"
        )

        post = {
            "title": item.get("title", ""),
            "url_id": item.get("urlId", ""),
            "published_date": published_date,
            "text": text,
        }
        posts.append(post)
        logger.info("  [%s] %s (%d chars)", published_date, post["title"], len(text))

    return posts


def ingest_all(force: bool = False) -> Path:
    """Fetch blog posts and cache to BLOG_CACHE_DIR/index.json.

    Parameters
    ----------
    force:
        If True, re-fetch even if the cache already exists.

    Returns
    -------
    Path to the cached index file.
    """
    BLOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if BLOG_INDEX_FILE.exists() and not force:
        logger.info("Blog cache hit: %s", BLOG_INDEX_FILE)
        return BLOG_INDEX_FILE

    posts = fetch_posts()
    BLOG_INDEX_FILE.write_text(
        json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d posts to %s", len(posts), BLOG_INDEX_FILE)
    return BLOG_INDEX_FILE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_all(force=False)

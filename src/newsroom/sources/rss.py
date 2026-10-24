"""Feed fetching and article text extraction.

Everything network-facing is injectable (``httpx.Client``) so tests run offline. Failures
never raise: the monitor records them in the audit trail and moves on.
"""

from __future__ import annotations

import html
import logging
import re
from calendar import timegm
from datetime import UTC, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import feedparser
import httpx

from newsroom.config import Settings, get_settings
from newsroom.domain import Article, SourceConfig, new_id

logger = logging.getLogger(__name__)

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    text = _SPACE.sub(" ", html.unescape(_TAG.sub(" ", value))).strip()
    return text or None


def _published(entry: feedparser.FeedParserDict) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(timegm(parsed), tz=UTC)
    except (OverflowError, ValueError):
        return None


def _client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )


def fetch_feed(
    source: SourceConfig,
    *,
    limit: int = 10,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> list[Article]:
    """Fetch and parse one feed. Returns [] on any failure or an empty feed."""

    settings = settings or get_settings()
    if not source.feed_url:
        return []

    owns_client = client is None
    http = client or _client(settings)
    try:
        response = http.get(source.feed_url)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
    except Exception as exc:
        logger.warning("Feed %s could not be fetched: %s", source.id, exc)
        return []
    finally:
        if owns_client:
            http.close()

    articles: list[Article] = []
    for entry in parsed.entries:
        url = (entry.get("link") or "").strip()
        title = _strip_html(entry.get("title"))
        if not url or not title:
            continue
        summary = _strip_html(entry.get("summary") or entry.get("description"))
        articles.append(
            Article(
                id=new_id("art"),
                source_id=source.id,
                source_name=source.name,
                url=url,
                title=title,
                summary=summary,
                language=source.language,
                published_at=_published(entry),
                official=source.official,
            )
        )
        if len(articles) >= limit:
            break
    return articles


_PAYWALL_FLAG = re.compile(r'isAccessibleForFree"?\s*:\s*"?false', re.IGNORECASE)
_PAYWALL_PHRASES = (
    "subscribe to continue",
    "subscribe to read",
    "subscribers only",
    "for subscribers",
    "become a subscriber",
    "log in to read",
    "tilaa jatkaaksesi",
    "tilaa lukeaksesi",
    "vain tilaajille",
    "tilaajille",
    "lue koko juttu tilaamalla",
    "kirjaudu sisään lukeaksesi",
    "jatka lukemista tilaamalla",
)


def looks_like_paywall(page_html: str | None, extracted: str | None) -> bool:
    """True if the page is a paywall stub. We never try to get around a paywall."""

    if page_html and _PAYWALL_FLAG.search(page_html):
        return True
    text = (extracted or "").casefold()
    if len(text) < 1500 and any(phrase in text for phrase in _PAYWALL_PHRASES):
        return True
    if not text and page_html:
        lowered = page_html.casefold()
        return any(phrase in lowered for phrase in _PAYWALL_PHRASES)
    return False


_ROBOTS_CACHE: dict[tuple[str, str], RobotFileParser | bool] = {}


def clear_robots_cache() -> None:
    """Test hook: the cache is process-wide, so tests must not inherit each other's hosts."""

    _ROBOTS_CACHE.clear()


def _robots_allows(url: str, settings: Settings, http: httpx.Client) -> bool:
    """Allowed only if robots.txt permits it, or does not exist (404/410).

    The parsed rules are cached per host and user agent: one article used to trigger one
    robots.txt request, so a feed with several stories paid for the same file repeatedly.
    An unreachable or erroring robots.txt means we do not crawl.
    """

    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return False
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    agent = settings.user_agent.split("/")[0].strip() or "*"
    key = (robots_url, agent)
    cached = _ROBOTS_CACHE.get(key)
    if cached is True:
        return True
    if cached is False:
        return False
    if isinstance(cached, RobotFileParser):
        return cached.can_fetch(agent, url)
    try:
        response = http.get(robots_url)
    except Exception:
        _ROBOTS_CACHE[key] = False
        return False
    if response.status_code in (404, 410):
        _ROBOTS_CACHE[key] = True
        return True
    if response.status_code != 200:
        _ROBOTS_CACHE[key] = False
        return False
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    if len(_ROBOTS_CACHE) > 256:
        _ROBOTS_CACHE.clear()
    _ROBOTS_CACHE[key] = parser
    return parser.can_fetch(agent, url)


def extract_content(
    url: str,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> str | None:
    """Extract the main article text, respecting robots.txt. None on any failure."""

    import trafilatura

    settings = settings or get_settings()
    owns_client = client is None
    http = client or _client(settings)
    try:
        if not _robots_allows(url, settings, http):
            logger.info("robots.txt does not allow fetching %s", url)
            return None
        response = http.get(url)
        response.raise_for_status()
        text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
        if looks_like_paywall(response.text, text):
            logger.info("%s looks paywalled; using headline and summary only", url)
            return None
        return text.strip() if text and text.strip() else None
    except Exception as exc:
        logger.warning("Content extraction failed for %s: %s", url, exc)
        return None
    finally:
        if owns_client:
            http.close()

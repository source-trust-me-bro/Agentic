"""Feed fetching and article text extraction.

Everything network-facing is injectable (``httpx.Client``) so tests run offline. Failures
never raise: the monitor records them in the audit trail and moves on.
"""

from __future__ import annotations

import html
import logging
import re
import time
from calendar import timegm
from datetime import UTC, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import feedparser
import httpx

from newsroom.config import Settings, get_settings
from newsroom.domain import Article, SourceConfig, new_id
from newsroom.sources.images import image_from_entry, image_from_page

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
                image_url=image_from_entry(entry, source.feed_url),
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


# robots.txt per host, read again after an hour: host -> (fetched at, parser; None = no file).
# City pages all live on one host, so this halves the requests of a cycle.
_ROBOTS: dict[str, tuple[float, RobotFileParser | None]] = {}
_ROBOTS_TTL_SECONDS = 3600


def _robots_allows(url: str, settings: Settings, http: httpx.Client) -> bool:
    """Allowed only if robots.txt permits it, or does not exist (404/410).

    An unreachable or erroring robots.txt means we do not crawl (and is not cached).
    """

    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return False
    host = f"{parts.scheme}://{parts.netloc}"
    cached = _ROBOTS.get(host)
    if cached is None or time.monotonic() - cached[0] > _ROBOTS_TTL_SECONDS:
        try:
            response = http.get(f"{host}/robots.txt")
        except Exception:
            return False
        if response.status_code in (404, 410):
            parser = None  # no robots.txt: allowed
        elif response.status_code != 200:
            return False
        else:
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
        _ROBOTS[host] = cached = (time.monotonic(), parser)
    parser = cached[1]
    agent = settings.user_agent.split("/")[0].strip() or "*"
    return True if parser is None else parser.can_fetch(agent, url)


def extract_page(
    url: str,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> tuple[str | None, str | None]:
    """(main article text, the page's og:image), respecting robots.txt. A paywalled page or
    any failure gives (None, None). Never call this for a metadata_only source."""

    import trafilatura

    settings = settings or get_settings()
    owns_client = client is None
    http = client or _client(settings)
    try:
        if not _robots_allows(url, settings, http):
            logger.info("robots.txt does not allow fetching %s", url)
            return None, None
        response = http.get(url)
        response.raise_for_status()
        text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
        if looks_like_paywall(response.text, text):
            logger.info("%s looks paywalled; using headline and summary only", url)
            return None, None
        text = text.strip() if text and text.strip() else None
        return text, image_from_page(response.text, url)
    except Exception as exc:
        logger.warning("Content extraction failed for %s: %s", url, exc)
        return None, None
    finally:
        if owns_client:
            http.close()


def extract_content(
    url: str,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> str | None:
    """Extract the main article text, respecting robots.txt. None on any failure."""

    return extract_page(url, settings=settings, client=client)[0]

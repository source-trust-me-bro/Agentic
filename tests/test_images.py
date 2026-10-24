"""The photo of a news item: from the feed first, from the article page second."""

from __future__ import annotations

import feedparser
import httpx

from newsroom.domain import SourceConfig
from newsroom.sources.images import image_from_entry, image_from_page
from newsroom.sources.rss import extract_page, fetch_feed


def _entry(item: str) -> feedparser.FeedParserDict:
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/" '
        'xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>T</title>'
        f"<item><title>News</title><link>https://news.test/a</link>{item}</item>"
        "</channel></rss>"
    )
    return feedparser.parse(feed.encode()).entries[0]


def test_media_content_largest_image_wins():
    entry = _entry(
        '<media:content url="https://news.test/small.jpg" medium="image" width="200"/>'
        '<media:content url="https://news.test/large.jpg" medium="image" width="1200"/>'
        '<media:content url="https://news.test/clip.mp4" medium="video" width="1920"/>'
    )
    assert image_from_entry(entry) == "https://news.test/large.jpg"


def test_image_enclosure_is_used():
    # Mun Oulu puts its photo in an enclosure on every item.
    entry = _entry('<enclosure url="https://munoulu.test/photo.jpg" type="image/jpeg" length="1"/>')
    assert image_from_entry(entry) == "https://munoulu.test/photo.jpg"


def test_img_in_summary_or_content_is_used_and_made_absolute():
    entry = _entry(
        "<description>&lt;p&gt;&lt;img src=\"/media/kuva.jpg\"&gt; Text&lt;/p&gt;</description>"
    )
    base = "https://ouka.test/news/feed"
    assert image_from_entry(entry, base) == "https://ouka.test/media/kuva.jpg"

    entry = _entry(
        "<content:encoded><![CDATA[<p>Text</p><img src='https://ouka.test/a.png'>]]>"
        "</content:encoded>"
    )
    assert image_from_entry(entry) == "https://ouka.test/a.png"


def test_logos_icons_and_svg_are_skipped():
    entry = _entry(
        '<enclosure url="https://news.test/brand/logo.png" type="image/png" length="1"/>'
        "<description>&lt;img src=\"https://news.test/icon.svg\"&gt;"
        "&lt;img src=\"https://news.test/photo.jpg\"&gt;</description>"
    )
    assert image_from_entry(entry) == "https://news.test/photo.jpg"
    svg = '<enclosure url="https://news.test/a.svg" type="image/svg+xml" length="1"/>'
    assert image_from_entry(_entry(svg)) is None
    assert image_from_entry(_entry("<description>No photo here.</description>")) is None


def test_og_image_in_either_attribute_order():
    first = '<head><meta property="og:image" content="https://site.test/og.jpg"></head>'
    second = "<head><meta content='/img/share.jpg' name='twitter:image'/></head>"
    assert image_from_page(first) == "https://site.test/og.jpg"
    assert image_from_page(second, "https://site.test/story") == "https://site.test/img/share.jpg"
    assert image_from_page('<meta property="og:image" content="https://site.test/logo.png">') is None
    assert image_from_page(None) is None


def test_fetch_feed_keeps_the_photo_link():
    feed = (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
        "<item><title>Bus news</title><link>https://feed.test/a</link>"
        '<enclosure url="https://feed.test/bus.jpg" type="image/jpeg" length="1"/></item>'
        "</channel></rss>"
    )
    http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=feed.encode()))
    )
    source = SourceConfig(id="t", name="T", feed_url="https://feed.test/rss")
    assert fetch_feed(source, client=http)[0].image_url == "https://feed.test/bus.jpg"


def test_extract_page_returns_text_and_og_image():
    page = (
        '<html><head><meta property="og:image" content="https://page.test/photo.jpg"></head>'
        "<body><article>"
        + "".join(
            f"<p>Paragraph {i}: the city council discussed winter transport timetables for "
            "residents and workers in the Oulu region, and explained what changes on Monday.</p>"
            for i in range(8)
        )
        + "</article></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=page)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    text, image = extract_page("https://page.test/story", client=http)
    assert text and "winter transport" in text
    assert image == "https://page.test/photo.jpg"


def test_robots_txt_is_fetched_once_per_host():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text="<html><body><p>short</p></body></html>")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    extract_page("https://cache.test/one", client=http)
    extract_page("https://cache.test/two", client=http)
    assert calls.count("/robots.txt") == 1

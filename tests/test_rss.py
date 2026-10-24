from __future__ import annotations

import httpx

from newsroom.domain import SourceConfig
from newsroom.sources.rss import extract_content, fetch_feed, looks_like_paywall

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item><title>First &amp; foremost</title><link>https://example.test/a</link>
<description>&lt;p&gt;A &lt;b&gt;summary&lt;/b&gt; here.&lt;/p&gt;</description>
<pubDate>Wed, 22 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>No link</title></item>
<item><title>Second</title><link>https://example.test/b</link></item>
<item><title>Third</title><link>https://example.test/c</link></item>
</channel></rss>"""

ARTICLE_HTML = (
    "<html><head><title>x</title></head><body><article><h1>Heading</h1>"
    + "".join(
        f"<p>Paragraph {i}: the city council discussed winter transport timetables for "
        "residents and workers in the Oulu region, and explained what changes on Monday.</p>"
        for i in range(8)
    )
    + "</article></body></html>"
)


def client(routes: dict[str, httpx.Response | Exception]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        outcome = routes.get(str(request.url))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome or httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


SOURCE = SourceConfig(
    id="s", name="Test Outlet", feed_url="https://feed.test/rss", language="fi", official=True
)


def test_fetch_feed_parses_entries_and_strips_html():
    http = client({"https://feed.test/rss": httpx.Response(200, content=FEED.encode())})
    articles = fetch_feed(SOURCE, limit=10, client=http)
    assert [a.url for a in articles] == [
        "https://example.test/a",
        "https://example.test/b",
        "https://example.test/c",
    ]
    first = articles[0]
    assert first.title == "First & foremost"
    assert first.summary == "A summary here."
    assert first.published_at is not None and first.published_at.tzinfo is not None
    assert first.official and first.source_name == "Test Outlet" and first.language == "fi"
    assert first.content is None  # no body extraction here


def test_fetch_feed_respects_limit():
    http = client({"https://feed.test/rss": httpx.Response(200, content=FEED.encode())})
    assert len(fetch_feed(SOURCE, limit=1, client=http)) == 1


def test_fetch_feed_never_raises():
    assert fetch_feed(SOURCE, client=client({"https://feed.test/rss": httpx.Response(500)})) == []
    assert (
        fetch_feed(SOURCE, client=client({"https://feed.test/rss": httpx.ConnectError("down")}))
        == []
    )
    no_url = SOURCE.model_copy(update={"feed_url": None})
    assert fetch_feed(no_url, client=client({})) == []


def test_extract_content_when_robots_missing():
    http = client({"https://example.test/page": httpx.Response(200, text=ARTICLE_HTML)})
    text = extract_content("https://example.test/page", client=http)  # robots.txt -> 404
    assert text and "winter transport" in text


def test_extract_content_blocked_by_robots_txt():
    http = client(
        {
            "https://example.test/robots.txt": httpx.Response(
                200, text="User-agent: *\nDisallow: /\n"
            ),
            "https://example.test/page": httpx.Response(200, text=ARTICLE_HTML),
        }
    )
    assert extract_content("https://example.test/page", client=http) is None


def test_extract_content_unreachable_robots_means_no_crawl():
    http = client(
        {
            "https://example.test/robots.txt": httpx.ConnectError("down"),
            "https://example.test/page": httpx.Response(200, text=ARTICLE_HTML),
        }
    )
    assert extract_content("https://example.test/page", client=http) is None
    http = client(
        {
            "https://example.test/robots.txt": httpx.Response(503),
            "https://example.test/page": httpx.Response(200, text=ARTICLE_HTML),
        }
    )
    assert extract_content("https://example.test/page", client=http) is None


def test_extract_content_returns_none_for_paywall_stub():
    stub = (
        '<html><head><script type="application/ld+json">{"isAccessibleForFree": false}'
        "</script></head><body><article><p>Tilaa jatkaaksesi lukemista.</p></article></body></html>"
    )
    http = client({"https://example.test/page": httpx.Response(200, text=stub)})
    assert extract_content("https://example.test/page", client=http) is None


def test_looks_like_paywall():
    assert looks_like_paywall('{"isAccessibleForFree":"False"}', None)
    assert looks_like_paywall("<p>x</p>", "Tilaa jatkaaksesi lukemista tästä jutusta.")
    assert not looks_like_paywall("<p>x</p>", "A normal article. " * 200)

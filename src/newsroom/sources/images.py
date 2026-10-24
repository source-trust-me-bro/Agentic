"""Find the photo that belongs to a news item: the feed first, the article page second.

Only a link to the publisher's own image is kept, never a copy. Readers see it with the outlet's
name, and it always links back to the original report.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

_IMG = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)
_META = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(r"([\w:-]+)\s*=\s*[\"']([^\"']*)[\"']")
_META_NAMES = ("og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src")
_SKIP = (".svg", ".gif", ".ico")  # logos, icons and tracking pixels
_SKIP_WORDS = ("logo", "icon", "pixel", "avatar", "placeholder", "1x1")


def _usable(url: str | None, base: str | None = None) -> str | None:
    if not url:
        return None
    url = url.strip()
    if base:
        url = urljoin(base, url)
    if not url.startswith(("http://", "https://")):
        return None
    path = url.lower().split("?", 1)[0]
    if path.endswith(_SKIP) or any(word in path for word in _SKIP_WORDS):
        return None
    return url


def _width(item: dict[str, Any]) -> int:
    try:
        return int(item.get("width") or 0)
    except (TypeError, ValueError):
        return 0


def image_from_entry(entry: Any, base: str | None = None) -> str | None:
    """The best image a feed entry carries: media:content, media:thumbnail, an image
    enclosure, or the first <img> in its HTML. Largest first."""

    candidates: list[tuple[int, str]] = []
    for key in ("media_content", "media_thumbnail"):
        for item in entry.get(key) or []:
            kind = f"{item.get('medium', '')} {item.get('type', '')}".strip()
            if kind and "image" not in kind:
                continue  # a video or audio file
            candidates.append((_width(item), item.get("url", "")))
    for link in list(entry.get("enclosures") or []) + list(entry.get("links") or []):
        if str(link.get("type", "")).startswith("image/"):
            candidates.append((_width(link), link.get("href") or link.get("url") or ""))
    image = entry.get("image")
    if isinstance(image, dict):
        candidates.append((0, image.get("href") or image.get("url") or ""))

    for _, url in sorted(candidates, key=lambda c: -c[0]):
        if usable := _usable(url, base):
            return usable

    html_parts = [entry.get("summary") or "", entry.get("description") or ""]
    html_parts += [c.get("value", "") for c in entry.get("content") or [] if isinstance(c, dict)]
    for html in html_parts:
        for match in _IMG.finditer(html):
            if usable := _usable(match.group(1), base):
                return usable
    return None


def image_from_page(html: str | None, base: str | None = None) -> str | None:
    """The page's own share image (og:image / twitter:image)."""

    for tag in _META.findall(html or ""):
        attrs = {k.lower(): v for k, v in _ATTR.findall(tag)}
        name = (attrs.get("property") or attrs.get("name") or "").lower()
        if name in _META_NAMES and (usable := _usable(attrs.get("content"), base)):
            return usable
    return None

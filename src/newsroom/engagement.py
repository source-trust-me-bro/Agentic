"""Following, likes, comments, shares and what they say about the news.

Members follow communities and official accounts; the personal feed, the "recommended to
follow" list, the top-news ranking and the dashboard all read the same stored records
(``Follow``, ``Reaction``, ``Comment``, ``Share``, ``View``), so nothing is counted anywhere but the
database. Members are pseudonymous references, as everywhere else in the newsroom.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, get_args
from urllib.parse import quote

from newsroom import guards
from newsroom.agents.distributor import _NON_HUMAN_REVIEWERS, DistributorAgent
from newsroom.domain import (
    Article,
    Channel,
    Comment,
    Community,
    Decision,
    Event,
    Follow,
    Member,
    OfficialAccount,
    Pitch,
    Publication,
    Reaction,
    Rendition,
    ReviewDecision,
    ReviewRole,
    Share,
    ShareTarget,
    SourceRef,
    Story,
    StoryStatus,
    Topic,
    TraceEvent,
    Translation,
    UserNeed,
    View,
    new_id,
    utcnow,
)
from newsroom.registry import load_official_accounts

if TYPE_CHECKING:
    from newsroom.service import Newsroom

# A share says more than a comment, a comment more than a like.
WEIGHTS = {"likes": 1, "comments": 3, "shares": 5}
SHARE_TARGETS: tuple[str, ...] = get_args(ShareTarget)
# Reads are counted beside the score, not in it: the editors see what people open and what
# they act on separately.
READS = ("views", "source_clicks")
COUNTED = (*READS, *WEIGHTS)
LIVE = (StoryStatus.PUBLISHED, StoryStatus.CORRECTED)


class EngagementError(ValueError):
    """The request is not allowed (unknown story, empty comment, software as an author)."""


def _stems(words: list[str]) -> set[str]:
    out: set[str] = set()
    for item in words:
        for word in item.replace("_", " ").casefold().split():
            if len(word) >= 4:
                out.add(word[:5])
    return out


class EngagementService:
    def __init__(self, newsroom: Newsroom) -> None:
        self.nr = newsroom
        self.store = newsroom.store
        self.accounts: dict[str, OfficialAccount] = {
            a.id: a for a in load_official_accounts()
        }
        for account in self.accounts.values():
            self.store.put(account)

    # ------------------------------------------------------------------ counting

    def _live_stories(self) -> dict[str, Story]:
        return {
            s.id: s for s in self.store.find(Story) if s.status in LIVE
        }

    def _counts(self, since: datetime | None = None) -> dict[str, Counter[str]]:
        """story_id -> views / source clicks / likes / comments / shares, from the records."""

        counts: dict[str, Counter[str]] = defaultdict(Counter)

        def fresh(doc: Any) -> bool:
            return since is None or doc.created_at >= since

        for reaction in self.store.find(Reaction):
            if fresh(reaction):
                counts[reaction.story_id]["likes"] += 1
        for comment in self.store.find(Comment):
            if comment.status == "visible" and fresh(comment):
                counts[comment.story_id]["comments"] += 1
        for share in self.store.find(Share):
            if fresh(share):
                counts[share.story_id]["shares"] += 1
        for view in self.store.find(View):
            if fresh(view):
                counts[view.story_id]["views" if view.kind == "open" else "source_clicks"] += 1
        return counts

    @staticmethod
    def _stat(counter: Counter[str] | None) -> dict[str, int]:
        c = counter or Counter()
        stat = {k: int(c.get(k, 0)) for k in COUNTED}
        stat["score"] = sum(WEIGHTS[k] * stat[k] for k in WEIGHTS)
        return stat

    def story_stats(self, story_id: str) -> dict[str, int]:
        return self._stat(self._counts().get(story_id))

    def all_stats(self, days: int | None = None) -> dict[str, dict[str, int]]:
        since = utcnow() - timedelta(days=days) if days else None
        return {sid: self._stat(c) for sid, c in self._counts(since).items()}

    # ---------------------------------------------------------------- reactions

    def _require_story(self, story_id: str) -> Story:
        story = self.store.get(Story, story_id)
        if story is None or story.status not in LIVE:
            raise KeyError(f"Story {story_id} is not published.")
        return story

    def like(self, ref: str, story_id: str, community_id: str | None = None) -> dict[str, int]:
        self._require_story(story_id)
        self.nr.member(ref)
        key = Reaction.key(ref, story_id)
        if self.store.get(Reaction, key) is None:
            self.store.put(
                Reaction(id=key, member_id=ref, story_id=story_id, community_id=community_id)
            )
        return self.story_stats(story_id)

    def view(
        self,
        ref: str,
        story_id: str,
        *,
        language: str | None = None,
        community_id: str | None = None,
    ) -> dict[str, int]:
        """A member opened the story. Once per member, story and day."""

        self._require_story(story_id)
        self.nr.member(ref)
        key = View.key(ref, story_id, "open", utcnow().date().isoformat())
        if self.store.get(View, key) is None:
            self.store.put(
                View(
                    id=key, member_id=ref, story_id=story_id, language=language,
                    community_id=community_id,
                )
            )  # fmt: skip
        return self.story_stats(story_id)

    def source_click(self, ref: str, story_id: str, url: str) -> dict[str, int]:
        """A member followed one of the story's source links. Once per member and link."""

        story = self._require_story(story_id)
        if url not in {s.url for s in story.sources}:
            raise EngagementError("That link is not one of this story's sources.")
        self.nr.member(ref)
        key = View.key(ref, story_id, "source", url)
        if self.store.get(View, key) is None:
            self.store.put(View(id=key, member_id=ref, story_id=story_id, kind="source", url=url))
        return self.story_stats(story_id)

    def unlike(self, ref: str, story_id: str) -> dict[str, int]:
        self.store.delete(Reaction, Reaction.key(ref, story_id))
        return self.story_stats(story_id)

    def comment(
        self,
        ref: str,
        story_id: str,
        text: str,
        *,
        language: str = "en",
        community_id: str | None = None,
    ) -> Comment:
        self._require_story(story_id)
        self.nr.member(ref)
        # Comments are shown to other members: contact details are masked before saving.
        clean = guards.minimise_personal_data(guards.sanitize(text, max_chars=1000)).strip()
        if not clean:
            raise EngagementError("A comment cannot be empty.")
        comment = Comment(
            id=new_id("cmt"),
            story_id=story_id,
            member_id=ref,
            community_id=community_id,
            text=clean,
            language=language,
        )
        self.store.put(comment)
        return comment

    def comments(self, story_id: str, limit: int = 50) -> list[dict[str, Any]]:
        out = []
        found = [c for c in self.store.find(Comment, story_id=story_id) if c.status == "visible"]
        for comment in sorted(found, key=lambda c: c.created_at, reverse=True)[:limit]:
            member = self.store.get(Member, comment.member_id)
            out.append(
                {
                    "id": comment.id,
                    "text": comment.text,
                    "language": comment.language,
                    "created_at": comment.created_at,
                    "author": (member.display_name if member else "") or "Community member",
                }
            )
        return out

    def hide_comment(self, comment_id: str, reviewer: str) -> Comment:
        comment = self.store.get(Comment, comment_id)
        if comment is None:
            raise KeyError(f"Unknown comment {comment_id}.")
        comment.status = "hidden"
        comment.hidden_by = self._human(reviewer)
        self.store.put(comment)
        return comment

    # ------------------------------------------------------------------ sharing

    def share_links(self, story_id: str) -> dict[str, str]:
        story = self._require_story(story_id)
        url = f"{self.nr.settings.public_base_url.rstrip('/')}/stories/{story_id}"
        u, text = quote(url, safe=""), quote(story.headline, safe="")
        return {
            "facebook": f"https://www.facebook.com/sharer/sharer.php?u={u}",
            "x": f"https://twitter.com/intent/tweet?url={u}&text={text}",
            "whatsapp": f"https://wa.me/?text={text}%20{u}",
            "telegram": f"https://t.me/share/url?url={u}&text={text}",
            "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={u}",
            "email": f"mailto:?subject={text}&body={u}",
            "copy_link": url,
        }

    def share(
        self, ref: str, story_id: str, platform: str, community_id: str | None = None
    ) -> dict[str, Any]:
        """Record that a member shared a story. The newsroom cannot see what the social
        network does next, so a share counts when the member presses the share button."""

        if platform not in SHARE_TARGETS:
            raise EngagementError(f"Unknown share target {platform}.")
        links = self.share_links(story_id)
        self.nr.member(ref)
        self.store.put(
            Share(
                id=new_id("shr"),
                story_id=story_id,
                member_id=ref,
                community_id=community_id,
                platform=platform,  # type: ignore[arg-type]
            )
        )
        return {"url": links[platform], "stats": self.story_stats(story_id)}

    # ---------------------------------------------------------------- following

    def _target(self, kind: str, target_id: str) -> Community | OfficialAccount:
        found = (self.nr.communities if kind == "community" else self.accounts).get(target_id)
        if found is None:
            raise KeyError(f"Unknown {kind} {target_id}.")
        return found

    def follow(self, ref: str, kind: str, target_id: str) -> Follow:
        if kind not in {"community", "official"}:
            raise EngagementError("kind must be 'community' or 'official'.")
        self._target(kind, target_id)
        member = self.nr.member(ref)
        key = Follow.key(ref, kind, target_id)
        existing = self.store.get(Follow, key)
        if existing is not None:
            return existing
        follow = Follow(id=key, member_id=ref, target_kind=kind, target_id=target_id)  # type: ignore[arg-type]
        self.store.put(follow)
        if kind == "community" and member.community_id is None:
            member.community_id = target_id  # the first community followed is the home one
            self.store.put(member)
        return follow

    def unfollow(self, ref: str, kind: str, target_id: str) -> bool:
        return self.store.delete(Follow, Follow.key(ref, kind, target_id))

    def set_language(
        self, ref: str, language: str, also_reads: list[str] | None = None
    ) -> Member:
        """The language the member reads in, and any others they also read."""

        member = self.nr.member(ref)
        member.language = language
        if also_reads is not None:
            member.reading_languages = [c for c in dict.fromkeys(also_reads) if c != language]
        self.store.put(member)
        return member

    def reading_languages(self, ref: str) -> list[str]:
        """Every language the member reads, their main one first."""

        member = self.nr.member(ref)
        return list(dict.fromkeys([member.language, *member.reading_languages]))

    def follower_counts(self) -> Counter[tuple[str, str]]:
        return Counter((f.target_kind, f.target_id) for f in self.store.find(Follow))

    def _card(
        self, kind: str, target: Community | OfficialAccount, followers: int
    ) -> dict[str, Any]:
        if isinstance(target, Community):
            languages = [lane.code for lane in target.languages]
        else:
            languages = list(target.languages)
        return {
            "kind": kind,
            "id": target.id,
            "name": target.name,
            "description": " ".join(target.description.split()),
            "languages": languages,
            "followers": followers,
        }

    def following(self, ref: str) -> list[dict[str, Any]]:
        counts = self.follower_counts()
        cards = []
        for follow in sorted(self.store.find(Follow, member_id=ref), key=lambda f: f.created_at):
            try:
                target = self._target(follow.target_kind, follow.target_id)
            except KeyError:
                continue
            cards.append(
                self._card(follow.target_kind, target, counts[(follow.target_kind, target.id)])
            )
        return cards

    def directory(self, ref: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """Every community and official account, most important first, marking what the
        member already follows."""

        counts = self.follower_counts()
        mine = (
            {(f.target_kind, f.target_id) for f in self.store.find(Follow, member_id=ref)}
            if ref
            else set()
        )
        communities = sorted(self.nr.communities.values(), key=lambda c: (c.priority, c.name))
        accounts = sorted(self.accounts.values(), key=lambda a: (a.priority, a.name))
        return {
            "communities": [
                {**self._card("community", c, counts[("community", c.id)]),
                 "following": ("community", c.id) in mine}
                for c in communities
            ],
            "official": [
                {**self._card("official", a, counts[("official", a.id)]),
                 "following": ("official", a.id) in mine}
                for a in accounts
            ],
        }

    # ------------------------------------------------------------ recommendations

    def recommend(self, ref: str, limit: int = 8, days: int = 14) -> list[dict[str, Any]]:
        """What to follow next, with the reason. A brand-new member gets suggestions from
        their language alone; each follow sharpens the list (regions, interests, languages)."""

        member = self.nr.member(ref)
        reads = set(self.reading_languages(ref))
        mine = {(f.target_kind, f.target_id) for f in self.store.find(Follow, member_id=ref)}
        followed = [
            self.nr.communities[t] for kind, t in mine if kind == "community" and t in self.nr.communities
        ]
        regions = {r.casefold() for c in followed for r in c.regions}
        interests = _stems([i for c in followed for i in c.interests])
        primary = {c.languages[0].code for c in followed if c.languages}
        counts = self.follower_counts()
        rankings = {r["id"]: r for r in self.community_rankings(days)}

        def common(priority: int, followers: int, engagement: int) -> float:
            return (
                max(0, 11 - priority) / 10
                + min(2.0, math.log1p(followers))
                + min(2.0, math.log1p(engagement))
            )

        scored: list[tuple[float, dict[str, Any]]] = []
        for community in self.nr.communities.values():
            if ("community", community.id) in mine:
                continue
            codes = {lane.code for lane in community.languages}
            if not codes & reads:
                continue  # nothing there the member could read
            reasons, score = [], 0.0
            if member.language in codes:
                score += 3
                reasons.append(f"Available in your language ({member.language})")
            if primary & codes:
                score += 1.5
                reasons.append("Shares a language with communities you follow")
            region_hits = {r.casefold() for r in community.regions} & regions
            if region_hits:
                score += min(2, len(region_hits))
                reasons.append("Near places you follow: " + ", ".join(sorted(region_hits)))
            topic_hits = _stems(community.interests) & interests
            if topic_hits:
                score += min(3, len(topic_hits)) * 0.5
                reasons.append("Covers topics you follow")
            followers = counts[("community", community.id)]
            if followers:
                reasons.append(f"{followers} follower{'s' if followers != 1 else ''}")
            score += common(
                community.priority, followers, rankings.get(community.id, {}).get("engagement", 0)
            )
            scored.append((score, {**self._card("community", community, followers), "reasons": reasons}))

        for account in self.accounts.values():
            if ("official", account.id) in mine:
                continue  # its stories reach the member in the languages our editions have
            reasons, score = [], 0.0
            if member.language in account.languages:
                score += 3
                reasons.append(f"Publishes in your language ({member.language})")
            region_hits = {r.casefold() for r in account.regions} & regions
            if region_hits:
                score += min(2, len(region_hits))
                reasons.append("Covers places you follow: " + ", ".join(sorted(region_hits)))
            if _stems(account.topics) & interests:
                score += 1.5
                reasons.append("Official source for topics you follow")
            followers = counts[("official", account.id)]
            score += common(account.priority, followers, 0)
            scored.append((score, {**self._card("official", account, followers), "reasons": reasons}))

        scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        return [{**card, "score": round(score, 2)} for score, card in scored[:limit]]

    # ------------------------------------------------------------------ the feed

    def _links(self, stories: Iterable[Story]) -> dict[str, tuple[set[str], list[str]]]:
        """For each story: the feeds it was reported from and the topics it belongs to.

        Read in three batches (pitches, articles, events), not three queries per story."""

        stories = list(stories)
        pitches = self.store.get_many(Pitch, (s.pitch_id for s in stories))
        articles = self.store.get_many(
            Article, (a for p in pitches.values() for a in p.article_ids)
        )
        events = self.store.get_many(Event, (a.event_id for a in articles.values() if a.event_id))
        out: dict[str, tuple[set[str], list[str]]] = {}
        for story in stories:
            pitch = pitches.get(story.pitch_id)
            sources: set[str] = set()
            topics: list[str] = []
            for article_id in pitch.article_ids if pitch else []:
                article = articles.get(article_id)
                if article is None:
                    continue
                sources.add(article.source_id)
                event = events.get(article.event_id) if article.event_id else None
                for topic_id in event.topic_ids if event else []:
                    if topic_id not in topics:
                        topics.append(topic_id)
            out[story.id] = (sources, topics)
        return out

    def story_topic_ids(self, story: Story) -> list[str]:
        return self._links([story])[story.id][1]

    def my_feed(self, ref: str, language: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Published stories from the communities and accounts the member follows, newest
        first. A member who follows nothing sees nothing, and a story that has no edition in
        a language the member reads is never shown (an English reader gets no Sámi story)."""

        wanted = [language] if language else self.reading_languages(ref)
        follows = self.store.find(Follow, member_id=ref)
        communities = [f.target_id for f in follows if f.target_kind == "community"]
        accounts = [
            self.accounts[f.target_id]
            for f in follows
            if f.target_kind == "official" and f.target_id in self.accounts
        ]
        sources_of = {a.id: set(a.source_ids) for a in accounts}

        stats = self.all_stats()
        liked = {r.story_id for r in self.store.find(Reaction, member_id=ref)}
        picked: dict[str, dict[str, Any]] = {}

        def add(item: dict[str, Any], why: str) -> None:
            entry = picked.get(item["story_id"])
            if entry is None:
                web = [r for r in item["renditions"] if r["channel"] == "web"]
                available = list(dict.fromkeys(r["language"] for r in web))
                chosen = next((code for code in wanted if code in available), None)
                if chosen is None:
                    return
                text = next((r["text"] for r in web if r["language"] == chosen), None)
                entry = {
                    **item,
                    "language": chosen,
                    "available_languages": available,
                    "text": text,
                    "because": [],
                    "stats": stats.get(item["story_id"]) or self._stat(None),
                    "liked_by_me": item["story_id"] in liked,
                }
                picked[item["story_id"]] = entry
            if entry is not None and why not in entry["because"]:
                entry["because"].append(why)

        cache: dict[str, list[dict[str, Any]]] = {}

        def items_for(community_id: str) -> list[dict[str, Any]]:
            if community_id not in cache:
                cache[community_id] = self.nr.public_feed(community_id)
            return cache[community_id]

        for community_id in communities:
            if community_id in self.nr.communities:
                name = self.nr.communities[community_id].name
                for item in items_for(community_id):
                    add(item, f"You follow {name}")
        if accounts:
            publications = self.store.find(Publication)
            stories = {
                s.id: s
                for s in self.store.get_many(Story, (p.story_id for p in publications)).values()
                if s.status in LIVE
            }
            links = self._links(stories.values())
            for publication in publications:
                story = stories.get(publication.story_id)
                if story is None:
                    continue
                story_sources = links[story.id][0]
                for account in accounts:
                    if sources_of[account.id] & story_sources:
                        for item in items_for(publication.community_id):
                            if item["story_id"] == story.id:
                                add(item, f"You follow {account.name}")
        feed = sorted(picked.values(), key=lambda i: i["published_at"], reverse=True)
        return feed[:limit]

    # --------------------------------------------------------------- what is hot

    def _rows(
        self,
        days: int | None,
        *,
        community_id: str | None = None,
        wanted: list[str] | None = None,
        stats: dict[str, dict[str, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Every published story with its engagement, best first. Built from a fixed number of
        queries however many stories there are."""

        stats = self.all_stats(days) if stats is None else stats
        stories = self._live_stories()
        where: dict[str, list[str]] = defaultdict(list)
        editions: dict[str, set[str]] = defaultdict(set)
        published: dict[str, datetime] = {}
        for pub in self.store.find(Publication):
            where[pub.story_id].append(pub.community_id)
            editions[pub.story_id] |= {r.language for r in pub.renditions if r.channel == "web"}
            published[pub.story_id] = max(
                pub.published_at, published.get(pub.story_id, pub.published_at)
            )

        chosen: list[tuple[Story, str | None]] = []
        for story_id, story in stories.items():
            if story_id not in where or (community_id and community_id not in where[story_id]):
                continue
            read_in = next((c for c in wanted or [] if c in editions[story_id]), None)
            if wanted and read_in is None:
                continue
            chosen.append((story, read_in))

        links = self._links(story for story, _ in chosen)
        # A headline in another language comes from the translation: one query for all of them.
        needs_translation = [s.id for s, lang in chosen if lang and lang != s.language]
        translated: dict[tuple[str, str], Translation] = {}
        for t in self.store.find_in(Translation, "story_id", needs_translation):
            if t.headline and t.body_md.strip():
                translated[(t.story_id, t.language)] = t

        rows = []
        for story, read_in in chosen:
            t = translated.get((story.id, read_in or ""))
            usable = t is not None and t.story_version == story.version
            rows.append(
                {
                    "story_id": story.id,
                    "headline": t.headline if usable and t else story.headline,
                    "language": read_in if usable else story.language,
                    "community_ids": sorted(set(where[story.id])),
                    "published_at": published[story.id],
                    "topic_ids": links[story.id][1],
                    **(stats.get(story.id) or self._stat(None)),
                }
            )
        rows.sort(key=lambda r: (r["score"], r["published_at"]), reverse=True)
        return rows

    def top_news(
        self,
        community_id: str | None = None,
        *,
        language: str | None = None,
        languages: list[str] | None = None,
        days: int | None = 7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Stories ranked by likes, comments and shares in the last ``days`` days. With
        ``languages`` (or one ``language``) only stories that have a web edition in one of
        them are listed, headlined in the first language they have."""

        wanted = languages or ([language] if language else None)
        return self._rows(days, community_id=community_id, wanted=wanted)[:limit]

    def topic_scores(
        self, days: int | None = 7, rows: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Engagement rolled up the topic tree: a story counts for its topics and for every
        topic above them, so the editors see which branches of the tree readers care about."""

        topics = {t.id: t for t in self.store.find(Topic) if t.status == "active"}
        total: dict[str, Counter[str]] = defaultdict(Counter)
        for row in self._rows(days) if rows is None else rows:
            seen: set[str] = set()
            for topic_id in row["topic_ids"]:
                while topic_id and topic_id in topics and topic_id not in seen:
                    seen.add(topic_id)
                    for key in (*COUNTED, "score"):
                        total[topic_id][key] += row[key]
                    total[topic_id]["stories"] += 1
                    topic_id = topics[topic_id].parent_id
        out = [
            {
                "topic_id": tid,
                "name": topics[tid].name,
                "parent_id": topics[tid].parent_id,
                **{k: int(c.get(k, 0)) for k in (*COUNTED, "score", "stories")},
            }
            for tid, c in total.items()
        ]
        return sorted(out, key=lambda r: (r["score"], r["views"], r["stories"]), reverse=True)

    def tree(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The whole information tree, parents before children, every topic with its reads
        and engagement (zero where nothing was read), so the editors see its full shape and
        which branches people open most."""

        topics = [t for t in self.store.find(Topic) if t.status == "active"]
        scores = {r["topic_id"]: r for r in self.topic_scores(rows=rows)}
        known = {t.id for t in topics}
        children: dict[str | None, list[Topic]] = defaultdict(list)
        for t in topics:
            children[t.parent_id if t.parent_id in known else None].append(t)
        zero = dict.fromkeys((*COUNTED, "score", "stories"), 0)
        out: list[dict[str, Any]] = []

        def walk(parent: str | None, depth: int) -> None:
            ranked = sorted(
                children[parent], key=lambda t: (-(scores.get(t.id) or zero)["views"], t.name)
            )
            for t in ranked:
                row = scores.get(t.id) or zero
                out.append(
                    {
                        "topic_id": t.id,
                        "name": t.name,
                        "parent_id": t.parent_id,
                        "depth": depth,
                        **{k: row[k] for k in zero},
                    }
                )
                walk(t.id, depth + 1)

        walk(None, 0)
        return out

    def community_rankings(
        self, days: int | None = 14, rows: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        counts = self.follower_counts()
        engagement: Counter[str] = Counter()
        stories: Counter[str] = Counter()
        for row in self._rows(days) if rows is None else rows:
            for cid in row["community_ids"]:
                engagement[cid] += row["score"]
                stories[cid] += 1
        out = [
            {
                "id": c.id,
                "name": c.name,
                "priority": c.priority,
                "followers": counts[("community", c.id)],
                "stories": stories[c.id],
                "engagement": engagement[c.id],
                "score": counts[("community", c.id)] + engagement[c.id],
            }
            for c in self.nr.communities.values()
        ]
        return sorted(out, key=lambda r: (-r["score"], r["priority"], r["name"]))

    def dashboard(self, days: int = 7, limit: int = 10) -> dict[str, Any]:
        """What the editors' dashboard shows: the most-read and most-engaged news, the busiest
        communities, the languages people read in and the information tree with its reads.
        The stories are read once and shared by all of them."""

        stats = self.all_stats(days)
        rows = self._rows(days, stats=stats)
        totals = {k: sum(s[k] for s in stats.values()) for k in COUNTED}
        follows = self.store.find(Follow)
        since = utcnow() - timedelta(days=days)
        by_language: Counter[str] = Counter(
            v.language or "unknown"
            for v in self.store.find(View)
            if v.kind == "open" and v.created_at >= since
        )
        most_read = sorted(
            (r for r in rows if r["views"]),
            key=lambda r: (r["views"], r["source_clicks"], r["score"]),
            reverse=True,
        )
        return {
            "generated_at": utcnow(),
            "days": days,
            "totals": {
                **totals,
                "follows": len(follows),
                "members_following": len({f.member_id for f in follows}),
            },
            "most_read": most_read[:limit],
            "top_stories": rows[:limit],
            "top_communities": self.community_rankings(days, rows)[:limit],
            "top_topics": self.topic_scores(days, rows)[:limit],
            "views_by_language": [
                {"language": lang, "views": n} for lang, n in by_language.most_common()
            ],
            "tree": self.tree(rows),
        }

    # ------------------------------------------------------------ editor posting

    @staticmethod
    def _human(name: str) -> str:
        name = (name or "").strip()
        if not name or name.casefold() in _NON_HUMAN_REVIEWERS:
            raise EngagementError("A post needs the name of a human editor.")
        return name

    def editor_post(
        self,
        *,
        editor: str,
        headline: str,
        standfirst: str,
        body_md: str,
        language: str = "en",
        community_ids: list[str] | None = None,
        destinations: list[str] | None = None,
        needs: list[str] | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """A named editor publishes their own story to chosen communities, or to all of them
        (``community_ids=None``). One story, one publication per community, so engagement is
        counted once however widely it is posted. The editor's name is the approval."""

        editor = self._human(editor)
        if not headline.strip() or not body_md.strip():
            raise EngagementError("A post needs a headline and a body.")
        targets = list(self.nr.communities) if community_ids is None else list(dict.fromkeys(community_ids))
        unknown = [c for c in targets if c not in self.nr.communities]
        if unknown or not targets:
            raise KeyError(f"Unknown community: {', '.join(unknown) or '(none chosen)'}.")
        targets.sort(key=lambda c: self.nr.communities[c].priority)
        dests = [d for d in (destinations or ["dashboard", "community"]) if d in {"dashboard", "community"}]
        if not dests:
            raise EngagementError("Choose dashboard, community or both.")
        need_list = [UserNeed(n) for n in (needs or [UserNeed.INFORMATION.value])]
        refs = [
            SourceRef(
                name=s["name"].strip(),
                url=s["url"].strip(),
                official=bool(s.get("official")),
                origin="editor",
                note=(s.get("note") or "").strip(),
            )
            for s in (sources or [])
            if (s.get("name") or "").strip()
        ]
        if not refs or any(not r.url.lower().startswith(("http://", "https://")) for r in refs):
            raise EngagementError("Every story needs at least one source with a web link.")
        text = f"{headline} {standfirst} {body_md}"
        risk, _ = guards.classify_risk(text, self.nr.ctx.policy)

        pitch = Pitch(
            id=new_id("pitch"),
            community_id=targets[0],
            article_ids=[],
            working_title=headline.strip()[:200],
            rationale=f"Written and posted by editor {editor}.",
            needs=need_list,
            risk_tier=risk,
            origin="editor",
            publish_to=dests,  # type: ignore[arg-type]
        )
        story = Story(
            id=new_id("story"),
            pitch_id=pitch.id,
            community_id=targets[0],
            language=language,
            headline=headline.strip(),
            standfirst=standfirst.strip(),
            body_md=body_md.strip(),
            needs=need_list,
            risk_tier=risk,
            sources=refs,
            ai_disclosure=f"Written by editor {editor}. Not machine-generated.",
            status=StoryStatus.PUBLISHED,
        )
        approval = ReviewDecision(
            id=new_id("rev"),
            story_id=story.id,
            story_version=story.version,
            role=ReviewRole.EDITOR_IN_CHIEF,
            reviewer=editor,
            decision=Decision.APPROVE,
            notes="Written and approved by the editor.",
        )
        self.store.put(pitch)
        self.store.put(story)
        self.store.put(approval)
        publications = []
        for community_id in targets:
            community = self.nr.communities[community_id]
            publication = Publication(
                id=new_id("pub"),
                story_id=story.id,
                story_version=story.version,
                community_id=community_id,
                destinations=dests,  # type: ignore[arg-type]
                renditions=self._renditions(story, community, editor),
                approval_ids=[approval.id],
            )
            self.store.put(publication)
            publications.append(publication.id)
        self.store.trace(
            TraceEvent(
                id=new_id("trace"),
                run_id=self.nr.ctx.run_id,
                agent="editor_post",
                step="posted",
                decision="published",
                reason=f"Posted by {editor} to {len(targets)} communities.",
                refs={"story_id": story.id, "communities": targets},
            )
        )
        return {
            "story_id": story.id,
            "community_ids": targets,
            "publication_ids": publications,
            "risk_tier": risk.value,
        }

    def _renditions(self, story: Story, community: Community, editor: str) -> list[Rendition]:
        link = next((s.url for s in story.sources if s.official), None) or (
            story.sources[0].url if story.sources else None
        )
        short = f"Posted by {editor}."
        out = []
        for channel in community.channels or [Channel(kind="web", label="Web feed")]:
            if channel.kind == "web":
                sources = "\n".join(f"- {s.name}: {s.url}" for s in story.sources)
                text = f"# {story.headline}\n\n{story.standfirst}\n\n{story.body_md}\n\n"
                if sources:
                    text += f"## Sources\n{sources}\n\n"
                text += story.ai_disclosure
            elif channel.kind == "messaging":
                text = DistributorAgent._messaging(
                    story.headline, story.standfirst, story.body_md, link, short, channel.max_chars
                )
            elif channel.kind == "audio":
                text = DistributorAgent._audio(
                    story.headline, story.standfirst, story.body_md, story.language, short
                )
            else:
                text = f"{story.headline}\n{story.standfirst}" + (f"\n{link}" if link else "")
            out.append(Rendition(channel=channel.kind, language=story.language, text=text, link=link))
        return out

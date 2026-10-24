"""Following, recommendations, likes, comments, shares, top news and editor posts."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from newsroom.api.app import create_app
from newsroom.domain import Comment, Follow, Publication, Reaction, Share, Story
from newsroom.engagement import EngagementError


def post(nr, headline="Bus timetables change", communities=None, language="en", **kw):
    return nr.engagement.editor_post(
        editor="Aino Editor",
        headline=headline,
        standfirst="What changes and when.",
        body_md="The new winter timetables start next week.",
        language=language,
        community_ids=communities,
        **kw,
    )


# ------------------------------------------------------------------ the communities


def test_ten_communities_cover_english_arabic_sami_and_finnish(newsroom):
    assert len(newsroom.communities) == 10
    languages = {lane.code for c in newsroom.communities.values() for lane in c.languages}
    assert {"en", "ar", "se", "fi"} <= languages
    assert newsroom.communities["new-in-finland"].priority == 1  # the most important first


def test_the_directory_lists_most_important_first_and_marks_follows(newsroom):
    newsroom.engagement.follow("u1", "community", "new-in-finland")
    newsroom.engagement.follow("u1", "official", "migri")

    directory = newsroom.engagement.directory("u1")

    priorities = [newsroom.communities[c["id"]].priority for c in directory["communities"]]
    assert priorities == sorted(priorities)
    assert {c["id"] for c in directory["communities"] if c["following"]} == {"new-in-finland"}
    assert {a["id"] for a in directory["official"] if a["following"]} == {"migri"}
    assert {a["id"] for a in directory["official"]} >= {
        "city-oulu", "migri", "police", "infofinland", "mun-oulu",
    }


# ---------------------------------------------------------------------- following


def test_a_member_can_follow_many_communities_and_following_twice_is_one_follow(newsroom):
    e = newsroom.engagement
    for cid in ["new-in-finland", "arabic-speakers-finland", "sapmi-north", "new-in-finland"]:
        e.follow("u1", "community", cid)
    e.follow("u1", "official", "infofinland")

    assert [c["id"] for c in e.following("u1")] == [
        "new-in-finland", "arabic-speakers-finland", "sapmi-north", "infofinland",
    ]
    assert len(newsroom.store.find(Follow, member_id="u1")) == 4
    assert newsroom.member("u1").community_id == "new-in-finland"  # the first one is home
    assert e.unfollow("u1", "community", "sapmi-north") is True
    assert len(e.following("u1")) == 3


def test_unknown_targets_are_refused(newsroom):
    with pytest.raises(KeyError):
        newsroom.engagement.follow("u1", "community", "nope")
    with pytest.raises(KeyError):
        newsroom.engagement.follow("u1", "official", "nope")
    with pytest.raises(EngagementError):
        newsroom.engagement.follow("u1", "group", "migri")


def test_recommendations_start_from_the_members_language_and_never_repeat_follows(newsroom):
    e = newsroom.engagement
    e.set_language("ar-user", "ar")

    first = e.recommend("ar-user", limit=30)

    ids = [r["id"] for r in first]
    assert ids.index("arabic-speakers-finland") < ids.index("workers-rights-finland")
    assert any("Arabic" in r or "ar" in r for r in next(r for r in first if r["id"] == "arabic-speakers-finland")["reasons"])
    assert {"infofinland", "migri", "city-oulu", "police", "mun-oulu"} <= set(ids)

    e.follow("ar-user", "community", "arabic-speakers-finland")
    e.follow("ar-user", "official", "infofinland")
    after = {r["id"] for r in e.recommend("ar-user", limit=30)}
    assert "arabic-speakers-finland" not in after and "infofinland" not in after


def test_following_oulu_communities_pulls_oulu_accounts_up(newsroom):
    e = newsroom.engagement
    def city(ref):
        return next(r for r in e.recommend(ref, limit=30) if r["id"] == "city-oulu")

    before = city("u2")
    e.follow("u2", "community", "oulu-international")
    after = city("u2")

    assert after["score"] > before["score"]
    assert any("Oulu" in text.title() or "oulu" in text for text in after["reasons"])


def test_popular_communities_rank_higher_for_new_members(newsroom):
    e = newsroom.engagement
    for n in range(5):
        e.follow(f"fan{n}", "community", "families-in-finland")

    ids = [r["id"] for r in e.recommend("fresh", limit=30)]

    assert ids.index("families-in-finland") < ids.index("international-students-finland")


# ------------------------------------------------------------------- editor posts


def test_an_editor_post_can_go_to_every_community(newsroom):
    result = post(newsroom)

    assert len(result["community_ids"]) == 10
    assert result["community_ids"][0] == "new-in-finland"  # most important first
    pubs = newsroom.store.find(Publication, story_id=result["story_id"])
    assert len(pubs) == 10
    story = newsroom.store.get(Story, result["story_id"])
    assert story.status.value == "published" and "Aino Editor" in story.ai_disclosure
    item = newsroom.public_feed("arabic-speakers-finland")[0]
    assert item["approvers"][0]["name"] == "Aino Editor"


def test_an_editor_post_can_go_to_chosen_communities_and_destinations(newsroom):
    result = post(newsroom, communities=["sapmi-north"], destinations=["dashboard"])

    assert result["community_ids"] == ["sapmi-north"]
    assert newsroom.public_feed("sapmi-north", "dashboard")
    assert newsroom.public_feed("sapmi-north", "community") == []
    assert newsroom.public_feed("lapland-seasonal") == []


def test_software_cannot_post_and_empty_posts_are_refused(newsroom):
    with pytest.raises(EngagementError):
        newsroom.engagement.editor_post(
            editor="bot", headline="h", standfirst="", body_md="b", community_ids=None
        )
    with pytest.raises(EngagementError):
        post(newsroom, headline="  ")
    with pytest.raises(KeyError):
        post(newsroom, communities=["nope"])


# ---------------------------------------------------------- likes, comments, shares


def test_likes_are_one_per_member_and_can_be_removed(newsroom):
    sid = post(newsroom)["story_id"]
    e = newsroom.engagement

    e.like("u1", sid)
    e.like("u1", sid)
    stats = e.like("u2", sid)

    assert stats["likes"] == 2 and len(newsroom.store.find(Reaction)) == 2
    assert e.unlike("u1", sid)["likes"] == 1


def test_only_published_stories_can_be_reacted_to(newsroom):
    with pytest.raises(KeyError):
        newsroom.engagement.like("u1", "story_missing")
    with pytest.raises(KeyError):
        newsroom.engagement.share_links("story_missing")


def test_comments_hide_contact_details_and_moderators_can_hide_them(newsroom):
    sid = post(newsroom)["story_id"]
    e = newsroom.engagement

    comment = e.comment("u1", sid, "Call me on +358 40 123 4567 or mail me@example.com")

    assert "me@example.com" not in comment.text and "123 4567" not in comment.text
    assert e.comments(sid)[0]["author"] == "Community member"
    with pytest.raises(EngagementError):
        e.comment("u1", sid, "   ")
    e.hide_comment(comment.id, "Aino Editor")
    assert e.comments(sid) == []
    assert e.story_stats(sid)["comments"] == 0
    with pytest.raises(EngagementError):
        e.hide_comment(comment.id, "bot")


def test_share_links_cover_the_social_networks_and_each_share_is_saved(newsroom):
    sid = post(newsroom)["story_id"]
    e = newsroom.engagement

    links = e.share_links(sid)
    assert set(links) == {"facebook", "x", "whatsapp", "telegram", "linkedin", "email", "copy_link"}
    assert links["facebook"].startswith("https://www.facebook.com/sharer/sharer.php?u=")
    assert links["copy_link"].endswith(f"/stories/{sid}")

    result = e.share("u1", sid, "facebook", "new-in-finland")
    e.share("u1", sid, "whatsapp")
    assert result["url"] == links["facebook"] and result["stats"]["shares"] == 1
    assert {s.platform for s in newsroom.store.find(Share)} == {"facebook", "whatsapp"}
    with pytest.raises(EngagementError):
        e.share("u1", sid, "myspace")


# ------------------------------------------------------------------ the feed


def test_the_feed_shows_followed_communities_in_the_members_language(newsroom):
    e = newsroom.engagement
    a = post(newsroom, "Arabic desk story", communities=["arabic-speakers-finland"])["story_id"]
    post(newsroom, "Sámi desk story", communities=["sapmi-north"])
    post(newsroom, "Lapland story", communities=["lapland-seasonal"])
    e.follow("u1", "community", "arabic-speakers-finland")
    e.follow("u1", "community", "sapmi-north")
    e.like("u1", a)

    feed = e.my_feed("u1")

    assert {i["headline"] for i in feed} == {"Arabic desk story", "Sámi desk story"}
    mine = next(i for i in feed if i["story_id"] == a)
    assert mine["liked_by_me"] is True and mine["stats"]["likes"] == 1
    assert mine["because"] == ["You follow Arabic speakers in Finland"]
    assert mine["language"] == "en" and mine["text"].startswith("# Arabic desk story")
    assert mine["available_languages"] == ["en"]  # no Arabic edition yet: falls back honestly


def test_a_story_posted_to_two_followed_communities_appears_once(newsroom):
    e = newsroom.engagement
    post(newsroom, communities=["new-in-finland", "arabic-speakers-finland"])
    e.follow("u1", "community", "new-in-finland")
    e.follow("u1", "community", "arabic-speakers-finland")

    feed = e.my_feed("u1")

    assert len(feed) == 1 and len(feed[0]["because"]) == 2


# --------------------------------------------------------------- what is hot


def test_top_news_weights_shares_over_comments_over_likes(newsroom):
    e = newsroom.engagement
    liked = post(newsroom, "Liked a lot", communities=["new-in-finland"])["story_id"]
    shared = post(newsroom, "Shared once", communities=["new-in-finland"])["story_id"]
    quiet = post(newsroom, "Quiet", communities=["sapmi-north"])["story_id"]
    for n in range(4):
        e.like(f"u{n}", liked)
    e.share("u9", shared, "x")
    e.share("u8", shared, "telegram")
    e.comment("u7", quiet, "Thanks!")

    top = e.top_news(days=7)

    assert [r["story_id"] for r in top] == [shared, liked, quiet]
    assert [r["score"] for r in top] == [10, 4, 3]
    assert [r["story_id"] for r in e.top_news("sapmi-north")] == [quiet]
    assert e.top_news(limit=1)[0]["story_id"] == shared


def test_the_dashboard_summarises_stories_communities_and_followers(newsroom):
    e = newsroom.engagement
    sid = post(newsroom, communities=["new-in-finland"])["story_id"]
    e.follow("u1", "community", "new-in-finland")
    e.follow("u2", "community", "new-in-finland")
    e.like("u1", sid)
    e.comment("u2", sid, "Useful")

    board = e.dashboard(days=7)

    assert board["totals"] == {
        "likes": 1, "comments": 1, "shares": 0, "follows": 2, "members_following": 2,
    }
    assert board["top_stories"][0]["story_id"] == sid
    top = board["top_communities"][0]
    assert top["id"] == "new-in-finland" and top["followers"] == 2 and top["engagement"] == 4


def test_the_export_for_the_reader_app_carries_each_stories_engagement(newsroom):
    e = newsroom.engagement
    sid = post(newsroom, communities=["new-in-finland"])["story_id"]
    e.like("u1", sid)

    story = next(s for s in newsroom.dashboard_export()["stories"] if s["story_id"] == sid)

    assert story["engagement"] == {"likes": 1, "comments": 0, "shares": 0, "score": 1}


def test_everything_is_saved_in_the_store(newsroom):
    sid = post(newsroom, communities=["new-in-finland"])["story_id"]
    e = newsroom.engagement
    e.follow("u1", "community", "new-in-finland")
    e.like("u1", sid)
    e.comment("u1", sid, "Hello")
    e.share("u1", sid, "linkedin")

    store = newsroom.store
    assert len(store.find(Follow)) == len(store.find(Reaction)) == len(store.find(Share)) == 1
    assert len(store.find(Comment)) == 1


# ------------------------------------------------------------------------- the API


def test_api_round_trip(newsroom):
    headers = {"X-User-Ref": "member-1"}
    with TestClient(create_app(newsroom)) as client:
        posted = client.post(
            "/staff/posts",
            json={
                "editor": "Aino Editor",
                "headline": "New bus lines",
                "body_md": "Details inside.",
                "community_ids": ["new-in-finland"],
            },
        )
        assert posted.status_code == 201
        sid = posted.json()["story_id"]

        assert client.post("/follows/me", json={"kind": "community", "target_id": "new-in-finland"}, headers=headers).status_code == 201
        assert client.post("/follows/me", json={"kind": "official", "target_id": "nope"}, headers=headers).status_code == 404
        recommended = client.get("/follows/recommended?limit=30", headers=headers).json()
        assert "new-in-finland" not in {r["id"] for r in recommended}
        assert {"migri", "infofinland"} <= {r["id"] for r in recommended}

        assert client.put(f"/stories/{sid}/like", headers=headers).json()["likes"] == 1
        assert client.post(f"/stories/{sid}/comments", json={"text": "Great"}, headers=headers).status_code == 201
        shared = client.post(f"/stories/{sid}/share", json={"platform": "facebook"}, headers=headers)
        assert shared.status_code == 201 and shared.json()["stats"]["score"] == 9
        assert client.post(f"/stories/{sid}/share", json={"platform": "myspace"}, headers=headers).status_code == 422
        assert client.put("/stories/story_missing/like", headers=headers).status_code == 404

        feed = client.get("/feed/me", headers=headers).json()
        assert [i["story_id"] for i in feed] == [sid] and feed[0]["stats"]["score"] == 9
        top = client.get("/top-news").json()
        assert top[0]["story_id"] == sid and top[0]["score"] == 9
        board = client.get("/dashboard/engagement").json()
        assert board["totals"]["follows"] == 1 and board["top_communities"][0]["id"] == "new-in-finland"

        assert client.delete("/follows/me/community/new-in-finland", headers=headers).json() == []
        assert client.get("/follows/me").status_code == 400  # needs the member reference
        assert client.post("/staff/posts", json={"editor": "bot", "headline": "x", "body_md": "y"}).status_code == 422


# ------------------------------------------- only what the member follows, in their language


def test_a_member_who_follows_nothing_sees_nothing(newsroom):
    post(newsroom)  # published to every community

    assert newsroom.engagement.my_feed("newcomer") == []


def test_an_english_reader_never_sees_a_sami_story_but_a_sami_reader_does(newsroom):
    e = newsroom.engagement
    post(newsroom, "Sámi-language story", communities=["sapmi-north"], language="se")
    post(newsroom, "English story", communities=["sapmi-north"], language="en")
    for ref in ("en-reader", "se-reader", "both-reader"):
        e.follow(ref, "community", "sapmi-north")
    e.set_language("se-reader", "se")
    e.set_language("both-reader", "en", also_reads=["se", "se", "en"])

    assert [i["headline"] for i in e.my_feed("en-reader")] == ["English story"]
    assert [i["headline"] for i in e.my_feed("se-reader")] == ["Sámi-language story"]
    assert {i["headline"] for i in e.my_feed("both-reader")} == {
        "English story", "Sámi-language story",
    }
    assert newsroom.member("both-reader").reading_languages == ["se"]


def test_top_news_and_community_feeds_can_be_limited_to_a_reading_language(newsroom):
    e = newsroom.engagement
    se = post(newsroom, "Sámi story", communities=["sapmi-north"], language="se")["story_id"]
    en = post(newsroom, "English story", communities=["sapmi-north"])["story_id"]
    e.like("u1", se)
    e.like("u1", en)

    assert {r["story_id"] for r in e.top_news(languages=["en"])} == {en}
    assert {r["story_id"] for r in e.top_news(languages=["se", "en"])} == {se, en}
    with TestClient(create_app(newsroom)) as client:
        feed = client.get("/feeds/sapmi-north", params={"language": "en"}).json()
        assert [i["story_id"] for i in feed] == [en]
        client.put("/members/me/language", json={"language": "en"}, headers={"X-User-Ref": "r1"})
        top = client.get("/top-news", headers={"X-User-Ref": "r1"}).json()
        assert [r["story_id"] for r in top] == [en]


def test_recommendations_skip_communities_the_member_cannot_read(newsroom):
    e = newsroom.engagement  # an English reader
    ids = {r["id"] for r in e.recommend("en-reader", limit=30)}
    assert "sapmi-north" not in ids  # Sámi and Finnish only
    assert "arabic-speakers-finland" in ids  # it also publishes in English
    assert "new-in-finland" in ids

    e.set_language("fi-reader", "fi")
    assert "sapmi-north" in {r["id"] for r in e.recommend("fi-reader", limit=30)}

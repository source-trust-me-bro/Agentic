"""Routes for following, views, likes, comments, shares, top news and editor posts.

Registered by ``create_app``. They use the same guards as the rest of the API: a staff key
for server-to-server calls and a pseudonymous ``X-User-Ref`` for the member.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from newsroom.engagement import EngagementError
from newsroom.service import Newsroom


class FollowIn(BaseModel):
    kind: Literal["community", "official"]
    target_id: str = Field(min_length=1)


class LanguageIn(BaseModel):
    language: str = Field(min_length=2, max_length=8)
    # Other languages the member also reads. Everything else is hidden from them.
    also_reads: list[str] | None = None


class CommentIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: str = "en"
    community_id: str | None = None


class ReactIn(BaseModel):
    community_id: str | None = None


class ShareIn(BaseModel):
    platform: str
    community_id: str | None = None


class HideIn(BaseModel):
    reviewer: str = Field(min_length=1)


class PostSourceIn(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    official: bool = False
    note: str = Field(default="", max_length=500)  # who the source is, for readers


class ViewIn(BaseModel):
    language: str | None = Field(default=None, max_length=12)
    community_id: str | None = None


class SourceClickIn(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


class EditorPostIn(BaseModel):
    editor: str = Field(min_length=1)
    headline: str = Field(min_length=1, max_length=200)
    standfirst: str = ""
    body_md: str = Field(min_length=1, max_length=20000)
    language: str = "en"
    # Empty or missing means every community.
    community_ids: list[str] | None = None
    destinations: list[Literal["dashboard", "community"]] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list)
    sources: list[PostSourceIn] = Field(default_factory=list)


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0] if exc.args else "Not found") from exc
    except EngagementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:  # e.g. an unknown need name
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def register(
    app: FastAPI,
    get_newsroom: Callable[[], Newsroom],
    user_ref: Callable[..., str],
    require_staff: Callable[..., None],
) -> None:
    staff = [Depends(require_staff)]

    @app.get("/directory", dependencies=staff)
    def directory(
        ref: str | None = Header(default=None, alias="X-User-Ref"),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, Any]:
        """Every community and official account, most important first, with follower counts."""

        return nr.engagement.directory(ref)

    @app.get("/follows/me", dependencies=staff)
    def my_follows(
        ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> list[dict[str, Any]]:
        return nr.engagement.following(ref)

    @app.post("/follows/me", status_code=201, dependencies=staff)
    def follow(
        body: FollowIn, ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> list[dict[str, Any]]:
        _call(nr.engagement.follow, ref, body.kind, body.target_id)
        return nr.engagement.following(ref)

    @app.delete("/follows/me/{kind}/{target_id}", dependencies=staff)
    def unfollow(
        kind: Literal["community", "official"],
        target_id: str,
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> list[dict[str, Any]]:
        nr.engagement.unfollow(ref, kind, target_id)
        return nr.engagement.following(ref)

    @app.get("/follows/recommended", dependencies=staff)
    def recommended_follows(
        limit: int = Query(default=8, ge=1, le=30),
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> list[dict[str, Any]]:
        return nr.engagement.recommend(ref, limit)

    @app.put("/members/me/language", dependencies=staff)
    def set_language(
        body: LanguageIn, ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        member = nr.engagement.set_language(ref, body.language, body.also_reads)
        return {"language": member.language, "reading_languages": member.reading_languages}

    @app.get("/feed/me", dependencies=staff)
    def my_feed(
        language: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> list[dict[str, Any]]:
        """Stories from every community and account the member follows."""

        return jsonable_encoder(nr.engagement.my_feed(ref, language, limit))

    @app.put("/stories/{story_id}/like", dependencies=staff)
    def like(
        story_id: str,
        body: ReactIn | None = None,
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, int]:
        return _call(nr.engagement.like, ref, story_id, body.community_id if body else None)

    @app.put("/stories/{story_id}/view", dependencies=staff)
    def view(
        story_id: str,
        body: ViewIn | None = None,
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, int]:
        """The member opened the story (counted once a day)."""

        body = body or ViewIn()
        return _call(
            nr.engagement.view, ref, story_id, language=body.language,
            community_id=body.community_id,
        )  # fmt: skip

    @app.post("/stories/{story_id}/source-click", dependencies=staff)
    def source_click(
        story_id: str,
        body: SourceClickIn,
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, int]:
        """The member followed one of the story's source links."""

        return _call(nr.engagement.source_click, ref, story_id, body.url)

    @app.delete("/stories/{story_id}/like", dependencies=staff)
    def unlike(
        story_id: str, ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, int]:
        return nr.engagement.unlike(ref, story_id)

    @app.get("/stories/{story_id}/comments", dependencies=staff)
    def story_comments(
        story_id: str, nr: Newsroom = Depends(get_newsroom)
    ) -> list[dict[str, Any]]:
        return jsonable_encoder(nr.engagement.comments(story_id))

    @app.post("/stories/{story_id}/comments", status_code=201, dependencies=staff)
    def add_comment(
        story_id: str,
        body: CommentIn,
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, Any]:
        comment = _call(
            nr.engagement.comment,
            ref,
            story_id,
            body.text,
            language=body.language,
            community_id=body.community_id,
        )
        return {"id": comment.id, "stats": nr.engagement.story_stats(story_id)}

    @app.get("/stories/{story_id}/share-links", dependencies=staff)
    def share_links(story_id: str, nr: Newsroom = Depends(get_newsroom)) -> dict[str, str]:
        return _call(nr.engagement.share_links, story_id)

    @app.post("/stories/{story_id}/share", status_code=201, dependencies=staff)
    def share(
        story_id: str,
        body: ShareIn,
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, Any]:
        return _call(nr.engagement.share, ref, story_id, body.platform, body.community_id)

    @app.get("/top-news", dependencies=staff)
    def top_news(
        community_id: str | None = Query(default=None),
        language: str | None = Query(default=None),
        days: int = Query(default=7, ge=1, le=365),
        limit: int = Query(default=10, ge=1, le=100),
        ref: str | None = Header(default=None, alias="X-User-Ref"),
        nr: Newsroom = Depends(get_newsroom),
    ) -> list[dict[str, Any]]:
        """Top news. For a member (X-User-Ref) only stories in a language they read."""

        if community_id and community_id not in nr.communities:
            raise HTTPException(status_code=404, detail="Unknown community.")
        reads = None if language or not ref else nr.engagement.reading_languages(ref)
        return jsonable_encoder(
            nr.engagement.top_news(
                community_id, language=language, languages=reads, days=days, limit=limit
            )
        )

    @app.get("/dashboard/engagement", dependencies=staff)
    def engagement_dashboard(
        days: int = Query(default=7, ge=1, le=365),
        limit: int = Query(default=10, ge=1, le=100),
        nr: Newsroom = Depends(get_newsroom),
    ) -> dict[str, Any]:
        """Most-read and top news, busiest communities, reading languages and the information
        tree with its reads, for the editors' dashboard."""

        return jsonable_encoder(nr.engagement.dashboard(days, limit))

    @app.post("/staff/posts", status_code=201, dependencies=staff)
    def editor_post(body: EditorPostIn, nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        """A named editor posts their own story to chosen communities, or to all of them."""

        data = body.model_dump()
        data["community_ids"] = data["community_ids"] or None
        return _call(nr.engagement.editor_post, **data)

    @app.post("/staff/comments/{comment_id}/hide", dependencies=staff)
    def hide_comment(
        comment_id: str, body: HideIn, nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, str]:
        comment = _call(nr.engagement.hide_comment, comment_id, body.reviewer)
        return {"id": comment.id, "status": comment.status}

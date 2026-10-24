"""HTTP API for the newsroom: run cycles, review stories, take community input, read feeds."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from newsroom.config import get_settings
from newsroom.domain import CommunityMessage, new_id
from newsroom.hitl import ReviewError
from newsroom.service import Newsroom
from newsroom.topics import get_tree


class CycleRequest(BaseModel):
    community_ids: list[str] | None = None
    use_fixtures: bool = False


class DecisionIn(BaseModel):
    role: str
    reviewer: str = Field(min_length=1)
    decision: str
    notes: str = ""


class ReviewRequest(BaseModel):
    decisions: list[DecisionIn]


class TranslationIn(BaseModel):
    language: str
    headline: str
    standfirst: str
    body_md: str
    reviewer: str = Field(min_length=1)


class SourceIn(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    official: bool = False
    reviewer: str = Field(min_length=1)


class MessageIn(BaseModel):
    community_id: str
    kind: str
    text: str = Field(min_length=1, max_length=4000)
    language: str = "en"
    author_ref: str = "anonymous"
    story_id: str | None = None


class JoinIn(BaseModel):
    community_id: str
    language: str | None = None


class CommunityRoleIn(BaseModel):
    user_ref: str = Field(min_length=1)
    community_id: str
    community_role: str
    review_roles: list[str] = Field(default_factory=list)
    review_languages: list[str] = Field(default_factory=list)
    display_name: str = ""


class LinkVoiceIn(BaseModel):
    user_ref: str = Field(min_length=1)
    community_id: str
    voice_id: str


class CorrectionIn(BaseModel):
    reviewer: str = Field(min_length=1)
    text: str = Field(min_length=1)


def create_app(newsroom: Newsroom | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.newsroom = newsroom or Newsroom(persistent_checkpoints=True)
        app.state.cycle_lock = threading.Lock()
        app.state.cycle_job = {"status": "idle"}
        stop = threading.Event()
        interval = settings.cycle_interval_minutes
        if interval > 0 and app.state.newsroom.ctx.llm.name != "fake":
            threading.Thread(
                target=_scheduler,
                args=(app.state.newsroom, stop, interval * 60),
                name="newsroom-scheduler",
                daemon=True,
            ).start()
        yield
        stop.set()

    app = FastAPI(
        title="Agentic newsroom",
        version="0.1.0",
        description=(
            "Agents find, select, produce and distribute community news. Humans approve "
            "everything that is published."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )

    def get_newsroom() -> Newsroom:
        return app.state.newsroom

    def user_ref(x_user_ref: str | None = Header(default=None)) -> str:
        """Pseudonymous account reference set by the web app's server, never by a browser."""

        if not x_user_ref or len(x_user_ref) > 128:
            raise HTTPException(status_code=400, detail="X-User-Ref header is required.")
        return x_user_ref

    def require_staff(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Staff API key required.")

    def _llm_status(nr: Newsroom) -> dict[str, Any]:
        """Whether the model can actually be used. Checked for real, then cached for 5 minutes."""

        cached = getattr(app.state, "llm_status", None)
        if cached and time.time() - cached["at"] < 300:
            return cached
        check = getattr(nr.ctx.llm, "check", None)
        ok, message = check() if callable(check) else (True, "no check needed")
        app.state.llm_status = {"at": time.time(), "ok": ok, "message": message}
        return app.state.llm_status

    @app.get("/health")
    def health(nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        llm_status = _llm_status(nr)
        return {
            "status": "ok",
            "llm": nr.ctx.llm.name,
            "llm_ok": llm_status["ok"],
            "llm_message": llm_status["message"],
            "communities": list(nr.communities),
            "sources": [s.id for s in nr.sources if s.enabled and s.feed_url],
        }

    @app.post("/cycle", dependencies=[Depends(require_staff)])
    def run_cycle(body: CycleRequest, nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        report = nr.run_cycle(body.community_ids, use_fixtures=body.use_fixtures)
        return asdict(report)

    def _scheduler(nr: Newsroom, stop: threading.Event, every_seconds: int) -> None:
        """The agents do the routine work on their own, so editors find stories ready."""

        while not stop.wait(every_seconds):
            if app.state.cycle_lock.acquire(blocking=False):
                _run_cycle_job(
                    nr, CycleRequest(), datetime.now(UTC).isoformat()
                )  # releases the lock when done

    def _run_cycle_job(nr: Newsroom, body: CycleRequest, started: str) -> None:
        job: dict[str, Any] = {"status": "running", "started_at": started}
        app.state.cycle_job = job
        try:
            report = nr.run_cycle(body.community_ids, use_fixtures=body.use_fixtures)
            job.update(
                status="done",
                finished_at=datetime.now(UTC).isoformat(),
                summary={
                    "stopped_reason": report.stopped_reason,
                    "articles_processed": report.articles_processed,
                    "articles_deferred": report.articles_deferred,
                    "new_articles": report.new_articles,
                    "events": report.events,
                    "pitches": report.pitches,
                    "stories": [
                        {"title": run.working_title, "status": run.status} for run in report.runs
                    ],
                },
            )
        except Exception as exc:  # a failed cycle must never take the service down
            job.update(
                status="failed",
                finished_at=datetime.now(UTC).isoformat(),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
        finally:
            app.state.cycle_lock.release()

    @app.post("/cycle/start", status_code=202, dependencies=[Depends(require_staff)])
    def start_cycle(body: CycleRequest, nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        """Start a cycle in the background and return at once. A real model takes minutes."""

        if not app.state.cycle_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A cycle is already running.")
        started = datetime.now(UTC).isoformat()
        app.state.cycle_job = {"status": "running", "started_at": started}
        threading.Thread(
            target=_run_cycle_job, args=(nr, body, started), name="newsroom-cycle", daemon=True
        ).start()
        return {"status": "running", "started_at": started}

    @app.post("/cycle/stop", dependencies=[Depends(require_staff)])
    def stop_cycle(nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        """Stop the running cycle at once. What it already saved stays saved."""

        if not app.state.cycle_lock.locked():
            return {"status": "idle"}
        nr.limits.cancel.set()
        return {"status": "stopping"}

    @app.get("/cycle/latest", dependencies=[Depends(require_staff)])
    def latest_cycle() -> dict[str, Any]:
        return dict(app.state.cycle_job)

    @app.get("/review-queue", dependencies=[Depends(require_staff)])
    def review_queue(nr: Newsroom = Depends(get_newsroom)) -> list[dict[str, Any]]:
        return nr.review_queue()

    @app.post("/review/{pitch_id}", dependencies=[Depends(require_staff)])
    def submit_review(
        pitch_id: str, body: ReviewRequest, nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        try:
            run = nr.submit_review(pitch_id, [d.model_dump() for d in body.decisions])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ReviewError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return asdict(run)

    @app.post("/stories/{story_id}/translation", dependencies=[Depends(require_staff)])
    def human_translation(
        story_id: str, body: TranslationIn, nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        try:
            translation = nr.submit_human_translation(story_id, **body.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return translation.model_dump(mode="json")

    @app.post("/stories/{story_id}/sources", dependencies=[Depends(require_staff)])
    def add_source(
        story_id: str, body: SourceIn, nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        try:
            story = nr.add_story_source(story_id, **body.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"story_id": story.id, "sources": [s.model_dump() for s in story.sources]}

    @app.get("/communities", dependencies=[Depends(require_staff)])
    def communities(nr: Newsroom = Depends(get_newsroom)) -> list[dict[str, Any]]:
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "regions": c.regions,
                "languages": [lane.model_dump() for lane in c.languages],
                "channels": [ch.model_dump() for ch in c.channels],
                "seasonal": c.lifecycle.seasonal,
                "trusted_voices": [{"name": v.name, "role": v.role} for v in c.trusted_voices],
            }
            for c in nr.communities.values()
        ]

    @app.get("/members/me", dependencies=[Depends(require_staff)])
    def my_membership(
        ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        member = nr.member(ref)
        return {
            "community_id": member.community_id,
            "language": member.language,
            "trusted_voice_id": member.trusted_voice_id,
            "community_role": member.community_role,
            "review_roles": [r.value for r in member.review_roles],
            "review_languages": member.review_languages,
        }

    @app.put("/members/me/community", dependencies=[Depends(require_staff)])
    def join_community(
        body: JoinIn, ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        try:
            member = nr.join_community(ref, body.community_id, body.language)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"community_id": member.community_id, "language": member.language}

    @app.post("/staff/community-role", dependencies=[Depends(require_staff)])
    def community_role(
        body: CommunityRoleIn, nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        try:
            member = nr.set_community_role(
                body.user_ref,
                body.community_id,
                community_role=body.community_role,
                review_roles=body.review_roles,
                review_languages=body.review_languages,
                display_name=body.display_name,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "community_id": member.community_id,
            "community_role": member.community_role,
            "review_roles": [r.value for r in member.review_roles],
            "review_languages": member.review_languages,
        }

    @app.post("/staff/link-voice", dependencies=[Depends(require_staff)])
    def link_voice(body: LinkVoiceIn, nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        try:
            member = nr.link_trusted_voice(body.user_ref, body.community_id, body.voice_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"community_id": member.community_id, "trusted_voice_id": member.trusted_voice_id}

    @app.post("/community/messages", status_code=201, dependencies=[Depends(require_staff)])
    def community_message(
        body: MessageIn, ref: str = Depends(user_ref), nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        if body.community_id not in nr.communities:
            raise HTTPException(status_code=404, detail="Unknown community.")
        if body.kind not in {"question", "tip", "correction", "local_knowledge"}:
            raise HTTPException(status_code=422, detail="Unknown message kind.")
        data = body.model_dump()
        data["author_ref"] = ref
        message = CommunityMessage(id=new_id("msg"), **data)
        triage = nr.community_message(message)
        return {"message_id": message.id, **triage.model_dump(mode="json")}

    @app.get("/community/messages/mine", dependencies=[Depends(require_staff)])
    def my_messages(
        community_id: str | None = Query(default=None),
        ref: str = Depends(user_ref),
        nr: Newsroom = Depends(get_newsroom),
    ) -> list[dict[str, Any]]:
        return [m.model_dump(mode="json") for m in nr.my_messages(ref, community_id)]

    @app.post("/corrections/{correction_id}/publish", dependencies=[Depends(require_staff)])
    def publish_correction(
        correction_id: str, body: CorrectionIn, nr: Newsroom = Depends(get_newsroom)
    ) -> dict[str, Any]:
        try:
            correction = nr.publish_correction(
                correction_id, reviewer=body.reviewer, text=body.text
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return correction.model_dump(mode="json")

    @app.get("/feeds/{community_id}", dependencies=[Depends(require_staff)])
    def feed(community_id: str, nr: Newsroom = Depends(get_newsroom)) -> list[dict[str, Any]]:
        if community_id not in nr.communities:
            raise HTTPException(status_code=404, detail="Unknown community.")
        return nr.public_feed(community_id)

    @app.get("/tree")
    def tree(nr: Newsroom = Depends(get_newsroom)) -> list[dict[str, Any]]:
        return [node.model_dump(mode="json") for node in get_tree(nr.store)]

    @app.get("/events/{event_id}")
    def event(event_id: str, nr: Newsroom = Depends(get_newsroom)) -> dict[str, Any]:
        try:
            return nr.architect.event_report(event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/traces", dependencies=[Depends(require_staff)])
    def traces(
        run_id: str | None = Query(default=None), nr: Newsroom = Depends(get_newsroom)
    ) -> list[dict[str, Any]]:
        return [t.model_dump(mode="json") for t in nr.store.traces(run_id)]

    return app


app = create_app()

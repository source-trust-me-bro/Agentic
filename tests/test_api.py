from __future__ import annotations

from fastapi.testclient import TestClient

from newsroom.api.app import create_app


def test_api_review_flow(newsroom):
    with TestClient(create_app(newsroom)) as client:
        assert client.get("/health").json()["llm"] == "fake"

        cycle = client.post(
            "/cycle", json={"community_ids": ["lapland-seasonal"], "use_fixtures": True}
        ).json()
        waiting = [run for run in cycle["runs"] if run["status"] == "awaiting_review"]
        assert waiting

        assert client.get("/feeds/lapland-seasonal").json() == []
        assert client.get("/feeds/unknown").status_code == 404

        # a rejected reviewer must not leave the paused story stuck
        # (checked below: the same story can still be reviewed afterwards)

        queue = client.get("/review-queue").json()
        assert queue and queue[0]["type"] == "human_review"

        pitch_id = waiting[0]["pitch_id"]
        bad = client.post(
            f"/review/{pitch_id}",
            json={
                "decisions": [{"role": "editor_in_chief", "reviewer": "bot", "decision": "approve"}]
            },
        )
        assert bad.status_code == 422

        good = client.post(
            f"/review/{pitch_id}",
            json={
                "decisions": [
                    {"role": "editor_in_chief", "reviewer": "Test editor", "decision": "reject"}
                ]
            },
        )
        assert good.status_code == 200
        assert good.json()["status"] == "rejected"


def test_api_requires_staff_key_when_configured(newsroom, monkeypatch):
    import newsroom.api.app as api_module

    monkeypatch.setattr(api_module.get_settings(), "api_key", "secret")
    with TestClient(create_app(newsroom)) as client:
        assert client.get("/review-queue").status_code == 401
        assert client.get("/review-queue", headers={"X-Api-Key": "secret"}).status_code == 200


def test_community_message_endpoint(newsroom):
    with TestClient(create_app(newsroom)) as client:
        ok = client.post(
            "/community/messages",
            headers={"X-User-Ref": "member-1"},
            json={
                "community_id": "oulu-international",
                "kind": "question",
                "text": "How do I get a tax card?",
            },
        )
        assert ok.status_code == 201
        mine = client.get("/community/messages/mine", headers={"X-User-Ref": "member-1"}).json()
        assert [m["kind"] for m in mine] == ["question"]
        assert (
            client.get("/community/messages/mine", headers={"X-User-Ref": "someone-else"}).json()
            == []
        )
        assert ok.json()["action"] in {"answer", "needs_human"}
        assert (
            client.post(
                "/community/messages",
                headers={"X-User-Ref": "member-1"},
                json={"community_id": "nowhere", "kind": "tip", "text": "x"},
            ).status_code
            == 404
        )


def test_a_cycle_can_run_in_the_background(newsroom):
    import time

    with TestClient(create_app(newsroom)) as client:
        assert client.get("/cycle/latest").json() == {"status": "idle"}

        started = client.post(
            "/cycle/start", json={"community_ids": ["lapland-seasonal"], "use_fixtures": True}
        )
        assert started.status_code == 202 and started.json()["status"] == "running"

        for _ in range(100):
            job = client.get("/cycle/latest").json()
            if job["status"] != "running":
                break
            time.sleep(0.1)

        assert job["status"] == "done", job
        assert job["summary"]["pitches"] >= 1
        assert job["summary"]["stories"][0]["status"] == "awaiting_review"


def test_only_one_cycle_runs_at_a_time(newsroom):
    with TestClient(create_app(newsroom)) as client:
        client.app.state.cycle_lock.acquire()  # a cycle is "running"
        try:
            busy = client.post("/cycle/start", json={"use_fixtures": True})
            assert busy.status_code == 409
        finally:
            client.app.state.cycle_lock.release()


def test_a_failing_cycle_is_reported_and_does_not_take_the_service_down(newsroom, monkeypatch):
    import time

    def boom(*args, **kwargs):
        raise RuntimeError("the model is unreachable")

    monkeypatch.setattr(newsroom, "run_cycle", boom)
    with TestClient(create_app(newsroom)) as client:
        client.post("/cycle/start", json={"use_fixtures": True})
        for _ in range(100):
            job = client.get("/cycle/latest").json()
            if job["status"] != "running":
                break
            time.sleep(0.1)

        assert job["status"] == "failed" and "unreachable" in job["error"]
        assert client.get("/health").status_code == 200
        assert client.post("/cycle/start", json={"use_fixtures": True}).status_code == 202

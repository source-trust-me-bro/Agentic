"""The editor can see what a cycle is doing: stages, stories and a percentage that only moves forward."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from newsroom.api.app import create_app
from newsroom.config import Settings
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.progress import STAGES, STORY_STEPS, CycleProgress
from newsroom.service import Newsroom
from newsroom.store.memory import MemoryStore


def stage(snapshot, key):
    return next(s for s in snapshot["stages"] if s["key"] == key)


def test_a_new_tracker_is_idle_with_every_stage_waiting():
    snap = CycleProgress().snapshot()

    assert snap["running"] is False and snap["percent"] == 0 and snap["outcome"] == ""
    assert [s["key"] for s in snap["stages"]] == [key for key, _, _ in STAGES]
    assert {s["status"] for s in snap["stages"]} == {"pending"}
    assert snap["stories"] == []


def test_stages_and_stories_move_the_percentage_forward_to_a_hundred():
    p = CycleProgress()
    p.start()
    seen = [p.snapshot()["percent"]]

    p.begin("collect", total=4)
    for _ in range(4):
        p.advance("collect")
        seen.append(p.snapshot()["percent"])
    p.finish_stage("collect")
    p.begin("group", total=2)
    p.advance("group", 2)
    p.finish_stage("group")
    p.finish_stage("select")
    p.begin("write", total=2)
    p.story_started("a", "Story A", "Community")
    p.story_started("b", "Story B", "Community")
    for step, _ in STORY_STEPS:
        p.story_step("a", step)
        seen.append(p.snapshot()["percent"])
    p.story_finished("a", "awaiting_review")
    p.story_finished("b", "awaiting_review")
    p.finish_stage("write")
    p.finish_stage("link")
    p.finish("done")
    seen.append(p.snapshot()["percent"])

    assert seen == sorted(seen)  # never goes backwards
    assert seen[0] == 0 and seen[-1] == 100
    final = p.snapshot()
    assert final["outcome"] == "done" and not final["running"]
    assert {s["status"] for s in final["stages"]} == {"done"}


def test_a_story_that_goes_round_again_never_moves_backwards():
    p = CycleProgress()
    p.start()
    p.story_started("a", "Story", "Community")
    p.story_step("a", "standards")
    p.story_step("a", "explain")  # a revision sends it back to writing

    story = p.snapshot()["stories"][0]
    assert story["step"] == "explain"
    assert story["steps_done"] == 3  # but what is done stays done


def test_a_stopped_or_failed_cycle_says_so_and_unfinished_stages_stop():
    p = CycleProgress()
    p.start()
    p.begin("write", total=3)
    p.finish("failed", "ValueError: boom")

    snap = p.snapshot()
    assert snap["outcome"] == "failed" and snap["note"] == "ValueError: boom"
    assert stage(snap, "write")["status"] == "stopped"
    assert snap["percent"] < 100


def test_the_tracker_is_safe_when_many_threads_report_at_once():
    p = CycleProgress()
    p.start()
    p.begin("collect", total=8000)

    def work():
        for _ in range(1000):
            p.advance("collect")

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert stage(p.snapshot(), "collect")["done"] == 8000


class SlowFake(FakeLLM):
    def generate(self, **kwargs):
        time.sleep(0.02)
        return super().generate(**kwargs)


def make_newsroom():
    return Newsroom(
        settings=Settings(llm_provider="fake", mongodb_uri=""),
        store=MemoryStore(),
        llm=SlowFake(collect_fake_handlers()),
        persistent_checkpoints=False,
    )


def test_a_real_cycle_reports_every_stage_and_story_while_it_runs():
    nr = make_newsroom()
    snapshots: list[dict] = []
    done = threading.Event()
    began: list[str] = []
    original_begin = nr.ctx.progress.begin

    def record(key, *args, **kwargs):
        began.append(key)
        return original_begin(key, *args, **kwargs)

    nr.ctx.progress.begin = record  # a stage can be over in microseconds: note each start

    def watch():
        while not done.is_set():
            snapshots.append(nr.ctx.progress.snapshot())
            time.sleep(0.01)

    watcher = threading.Thread(target=watch)
    watcher.start()
    report = nr.run_cycle(None, use_fixtures=True)
    done.set()
    watcher.join()
    snapshots.append(nr.ctx.progress.snapshot())

    percents = [s["percent"] for s in snapshots if s["running"] or s["outcome"]]
    assert percents == sorted(percents)  # a bar that only moves forward
    assert len(set(percents)) > 5  # and really moves in between, not just 0 then 100
    final = snapshots[-1]
    assert final["outcome"] == "done" and final["percent"] == 100 and not final["running"]
    assert {s["status"] for s in final["stages"]} == {"done"}
    assert len(final["stories"]) == len(report.runs) == stage(final, "write")["total"]
    assert {s["step"] for s in final["stories"]} == {"review"}  # all reached the human gate
    assert all(s["status"] == "awaiting_review" for s in final["stories"])
    assert {"collect", "group", "select", "write", "link"} <= set(began)  # every stage began
    seen_steps = {st["step"] for s in snapshots for st in s["stories"]}
    assert {"investigate", "explain", "translate", "standards", "review"} <= seen_steps


def test_a_cycle_that_fails_is_reported_as_failed(monkeypatch):
    nr = make_newsroom()

    def boom(*args, **kwargs):
        raise RuntimeError("feeds exploded")

    monkeypatch.setattr(nr.monitor, "scan", boom)

    try:
        nr.run_cycle(None, use_fixtures=True)
    except RuntimeError:
        pass

    snap = nr.ctx.progress.snapshot()
    assert snap["outcome"] == "failed" and "feeds exploded" in snap["note"] and not snap["running"]


def test_the_api_serves_the_progress_with_the_cycle_status():
    nr = make_newsroom()
    with TestClient(create_app(nr)) as client:
        before = client.get("/cycle/latest").json()
        assert before["progress"]["running"] is False

        nr.run_cycle(["new-in-finland"], use_fixtures=True)
        after = client.get("/cycle/latest").json()["progress"]

    assert after["percent"] == 100 and after["outcome"] == "done"
    assert [s["label"] for s in after["stages"]][0] == "Collecting news"
    assert after["story_steps"][-1]["label"] == "Ready for a person to review"


# ------------------------------------------------------------- the advanced view


def test_the_live_feed_lists_what_just_happened_newest_first_and_stays_short():
    p = CycleProgress()
    p.start()
    p.begin("collect", total=2)
    p.advance("collect", detail="Read Yle News")
    p.story_started("a", "Winter tyres", "Lapland workers")
    p.story_step("a", "explain")
    for n in range(200):
        p.log("reporter", f"Checked claim {n}")

    feed = p.snapshot()["activity"]

    assert len(feed) <= 25 and feed[0]["text"] == "Checked claim 199"  # newest first, short
    assert all({"t", "agent", "text", "level"} <= set(item) for item in feed)

    q = CycleProgress()
    q.start()
    q.begin("collect", total=2)
    q.advance("collect", detail="Read Yle News")
    q.story_started("a", "Winter tyres", "Lapland workers")
    q.story_step("a", "explain")
    texts = [item["text"] for item in q.snapshot()["activity"]]
    assert any("Read Yle News" in t for t in texts)
    assert any("Winter tyres" in t and "writing the story" in t for t in texts)
    assert texts[-1] == "The cycle started"


def test_the_time_left_appears_once_there_is_something_to_go_on():
    p = CycleProgress()
    assert p.snapshot()["eta_seconds"] is None  # nothing running
    p.start()
    assert p.snapshot()["eta_seconds"] is None  # 0 percent: too early to say

    p.begin("collect", total=1)
    p.finish_stage("collect")
    p.begin("group", total=1)
    p.finish_stage("group")  # 25 percent
    time.sleep(0.05)
    running = p.snapshot()
    assert running["percent"] == 25 and running["eta_seconds"] is not None
    assert running["eta_seconds"] >= 0

    p.finish("done")
    assert p.snapshot()["eta_seconds"] is None  # finished: nothing left


def test_each_stage_reports_how_long_it_took():
    p = CycleProgress()
    p.start()
    assert stage(p.snapshot(), "collect")["seconds"] is None  # not started
    p.begin("collect", total=1)
    time.sleep(0.15)  # seconds are shown to one decimal place
    p.finish_stage("collect")

    seconds = stage(p.snapshot(), "collect")["seconds"]
    assert seconds is not None and seconds >= 0.1
    assert "_t0" not in stage(p.snapshot(), "collect")  # internals stay internal


def test_model_usage_counts_calls_in_flight_time_and_tasks():
    from newsroom.limits import CycleLimits
    from newsroom.llm.budget import BudgetedLLM

    class Inner:
        name = "fake"

        def generate(self, **kwargs):
            time.sleep(0.03)
            return "ok"

    limits = CycleLimits(max_model_calls=100, max_seconds=60)
    limits.start()
    llm = BudgetedLLM(Inner(), limits, max_concurrent=4)
    before = llm.stats()
    assert before["calls"] == 0 and before["avg_seconds"] is None and before["in_flight"] == 0

    from newsroom.parallel import run_parallel

    run_parallel(
        lambda n: llm.generate(task="translator.translate_story" if n % 2 else "assess_evidence",
                               system="", prompt="", schema=object),
        range(12),
        workers=6,
    )
    after = llm.stats()

    assert after["calls"] == 12 and after["budget"] == 100 and after["in_flight"] == 0
    assert 1 < after["peak_in_flight"] <= 4 and after["max_concurrent"] == 4
    assert after["avg_seconds"] >= 0.03
    assert after["by_task"] == {"translator.translate_story": 6, "assess_evidence": 6}
    llm.reset_stats()
    assert llm.stats()["by_task"] == {} and llm.stats()["avg_seconds"] is None


def test_a_real_cycle_reports_languages_usage_and_a_live_feed():
    nr = make_newsroom()
    nr.run_cycle(["new-in-finland"], use_fixtures=True)

    snap = nr.ctx.progress.snapshot()

    assert snap["stories"], "the cycle wrote at least one story"
    for story in snap["stories"]:
        assert story["languages_total"] > 0
        assert story["languages_done"] == story["languages_total"]  # every language was counted
    usage = snap["usage"]
    assert usage["calls"] > 0 and usage["budget"] > usage["calls"] and usage["in_flight"] == 0
    assert "translator.translate_story" in usage["by_task"]
    agents = {item["agent"] for item in snap["activity"]}
    assert {"newsroom", "translator", "editor"} <= agents
    assert all(stage_["seconds"] is not None for stage_ in snap["stages"])

"""Offline walkthrough: fixtures in, human-approved community stories out.

The reviewers in this demo are SIMULATED stand-ins so the whole path can be shown in one
run. In a real newsroom each decision is made by a named person. The demo uses the same
code path and the same rules, so nothing is publishable without every required role.
"""

from __future__ import annotations

from newsroom.domain import Community, ReviewRole
from newsroom.service import Newsroom, StoryRun
from newsroom.topics import get_tree

LINE = "-" * 78


def _h(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def _reviewer_for(community: Community, role: ReviewRole, languages: set[str]) -> str:
    if role is ReviewRole.EDITOR_IN_CHIEF:
        return "Editor-in-chief (simulated for the demo)"
    for voice in community.trusted_voices:
        if role not in voice.can_review:
            continue
        if (
            role is ReviewRole.LANGUAGE_REVIEWER
            and languages
            and not languages & set(voice.languages)
        ):
            continue
        return voice.id
    return ""


def _print_tree(nodes, depth: int = 0) -> None:
    for node in nodes:
        events = [e for e in node.events if e.status != "merged"]
        if events or node.children:
            marker = f"  ({len(events)} event{'s' if len(events) != 1 else ''})" if events else ""
            print(f"{'  ' * depth}- {node.topic.name}{marker}")
            for event in events:
                outlets = ", ".join(sorted(set(event.source_names)))
                print(f"{'  ' * (depth + 1)}* {event.title}  [{outlets}]")
        _print_tree(node.children, depth + 1)


def run_demo(community_id: str = "lapland-seasonal", approve: bool = True) -> int:
    newsroom = Newsroom()
    community = newsroom.communities.get(community_id)
    if community is None:
        print(f"Unknown community '{community_id}'. Choose from: {', '.join(newsroom.communities)}")
        return 2

    _h(f"1. Cycle for: {community.name}   (model layer: {newsroom.ctx.llm.name})")
    report = newsroom.run_cycle([community_id], use_fixtures=True)
    print(
        f"New articles: {report.new_articles}   Events: {report.events}   Pitches: {report.pitches}"
    )
    if report.flagged_articles:
        print(
            f"Articles with instruction-like text (kept away from agents): {report.flagged_articles}"
        )

    _h("2. Same news from several outlets became one event in the tree")
    for event_id in report.event_ids:
        info = newsroom.architect.event_report(event_id)
        if info["article_count"] > 1:
            print(f"* {info['title']}")
            print(
                f"    independent outlets: {info['independent_outlet_count']}   languages: {info['languages']}"
            )
            for outlet in info["outlets"]:
                access = (
                    "headline only (paywalled)"
                    if outlet["access"] == "metadata_only"
                    else "full text"
                )
                print(f"    - {outlet['name']} [{outlet['language']}] {access}")
    print()
    _print_tree(get_tree(newsroom.store))

    _h("3. Stories prepared by the agents (nothing is published yet)")
    for run in report.runs:
        print(f"[{run.status}] {run.working_title}")
        if run.pending:
            print(f"    risk: {run.pending['risk_tier']}   needs: {run.pending['missing_roles']}")
            print(f"    translations: {run.pending['translations']}")

    if not approve:
        print("\nStopped at the human gate (--no-approve). Use the API to submit reviews.")
        return 0

    _h("4. Humans review (SIMULATED reviewers; real ones are named staff and trusted voices)")
    runs: list[StoryRun] = list(report.runs)
    for run in runs:
        for _ in range(4):
            if not run.pending:
                break
            roles = [ReviewRole(r) for r in run.pending["missing_roles"]]
            languages = {code for code in run.pending["translations"]}
            decisions = []
            for role in roles:
                reviewer = _reviewer_for(community, role, languages)
                if not reviewer:
                    print(f"    no qualified reviewer for {role.value}; the story waits")
                    break
                decisions.append(
                    {"role": role.value, "reviewer": reviewer, "decision": "approve", "notes": "ok"}
                )
            if not decisions:
                break
            run = newsroom.submit_review(run.pitch_id, decisions)
        print(f"[{run.status}] {run.working_title}")

    _h("5. What the community receives (approved renditions)")
    for publication in newsroom.feed(community_id):
        for rendition in publication.renditions:
            if rendition.channel in {"messaging", "audio"} and rendition.language == "en":
                print(f"[{rendition.channel}] {rendition.text}\n")
    if not newsroom.feed(community_id):
        print("Nothing was published: every story still needs a human.")

    _h("6. Community desk: a tip, a question and a correction")
    from newsroom.domain import CommunityMessage, new_id

    samples = [
        CommunityMessage(
            id=new_id("msg"),
            community_id=community_id,
            kind="tip",
            text="Our staff housing in Levi has no heating this week. Is anyone else affected?",
        ),
        CommunityMessage(
            id=new_id("msg"),
            community_id=community_id,
            kind="question",
            text="Do I need a tax card before my first pay day?",
        ),
    ]
    for message in samples:
        triage = newsroom.community_message(message)
        print(f"[{message.kind}] -> {triage.action}: {triage.reason}")

    _h("7. Audit trail")
    for event in newsroom.store.traces(report.run_id)[-12:]:
        print(f"{event.agent:>20} · {event.step:<14} {event.decision:<14} {event.reason[:60]}")
    return 0

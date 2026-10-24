from __future__ import annotations

from _b_helpers import community, make_ctx, write_story
from newsroom.agents.translator import (
    TranslationDraft,
    TranslatorAgent,
    compare_texts,
    extract_numbers,
)
from newsroom.domain import LanguageLane, RiskTier, Translation
from newsroom.llm.base import LLMError


def _sami_lane() -> LanguageLane:
    return community("sapmi-north").lane("se")  # type: ignore[return-value]


def test_north_sami_is_never_machine_translated():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx, "sapmi-north", title="Sámi daycare gets funding")
    translation = TranslatorAgent(ctx).translate(story, _sami_lane(), comm)

    assert translation.status == "needs_human_translation"
    assert translation.body_md == "" and translation.headline == ""
    assert translation.machine_translated is False
    assert not any(task.startswith("translator.") for task, _ in ctx.llm.calls)  # type: ignore[attr-defined]
    assert ctx.store.get(Translation, translation.id) is not None


def test_tier_a_language_gets_a_labelled_machine_draft():
    ctx = make_ctx()
    articles_title = "Winter bus timetables"
    story, _, comm, _ = write_story(ctx, "oulu-international", title=articles_title)
    story.risk_tier = RiskTier.LOW
    lane = comm.lane("uk")
    translation = TranslatorAgent(ctx).translate(story, lane, comm)  # type: ignore[arg-type]

    assert translation.status == "machine_draft" and translation.machine_translated
    assert translation.body_md.startswith("[offline stand-in · en→uk]")
    assert translation.backtranslation_flags == []  # low-risk tier A: no extra back-check


def test_changed_number_is_flagged_by_back_translation():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx)  # permit story is HIGH risk
    assert story.risk_tier is RiskTier.HIGH

    def drifting(payload):
        return TranslationDraft(
            headline=payload["headline"].replace("45", "54"),
            standfirst=payload["standfirst"],
            body_md=payload["body_md"].replace("45", "54"),
        )

    ctx.llm.register("translator.backtranslate", drifting)
    translation = TranslatorAgent(ctx).translate(story, comm.lane("uk"), comm)  # type: ignore[arg-type]

    assert translation.status == "machine_draft"
    assert any(
        "number 45" in f and "back-translation" in f for f in translation.backtranslation_flags
    )


def test_missing_protected_term_is_flagged():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx)

    def forgetful(payload):
        strip = lambda text: text.replace("Migri", "the agency")  # noqa: E731
        return TranslationDraft(
            headline=strip(payload["headline"]),
            standfirst=strip(payload["standfirst"]),
            body_md=strip(payload["body_md"]),
        )

    ctx.llm.register("translator.translate_story", forgetful)
    translation = TranslatorAgent(ctx).translate(story, comm.lane("uk"), comm)  # type: ignore[arg-type]

    assert "Migri" in translation.protected_terms
    assert any("protected term 'Migri'" in f for f in translation.backtranslation_flags)


def test_translation_is_never_marked_human_reviewed():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx)
    translation = TranslatorAgent(ctx).translate(story, comm.lane("ar"), comm)  # type: ignore[arg-type]
    assert translation.status != "human_reviewed"


def test_provider_failure_hands_over_to_a_human():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx)

    def broken(payload):
        raise LLMError("offline")

    ctx.llm.register("translator.translate_story", broken)
    translation = TranslatorAgent(ctx).translate(story, comm.lane("uk"), comm)  # type: ignore[arg-type]

    assert translation.status == "needs_human_translation"
    assert translation.body_md == ""


def test_failed_back_translation_check_is_flagged_not_skipped():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx)

    def broken(payload):
        raise LLMError("offline")

    ctx.llm.register("translator.backtranslate", broken)
    translation = TranslatorAgent(ctx).translate(story, comm.lane("uk"), comm)  # type: ignore[arg-type]
    assert any("could not run" in f for f in translation.backtranslation_flags)


def test_source_language_lane_needs_no_translation():
    ctx = make_ctx()
    story, _, comm, _ = write_story(ctx)
    translation = TranslatorAgent(ctx).translate(story, comm.lane("en"), comm)  # type: ignore[arg-type]
    assert translation.body_md == story.body_md and translation.machine_translated is False


def test_number_helpers():
    assert extract_numbers("45 days, 4 €, 08:00, 42–18, 1 000") >= {
        "45",
        "4",
        "0800",
        "42",
        "18",
        "1000",
    }
    assert extract_numbers("٤٥ يومًا") == {"45"}  # Arabic-Indic digits
    flags = compare_texts("Pay 45 days Migri", "Pay 54 days", ["Migri"], "x")
    assert any("45" in f for f in flags) and any("54" in f for f in flags)
    assert any("Migri" in f for f in flags)

"""A story is filed under the language it is really in, and machine-written Sámi never goes out."""

from __future__ import annotations

from _b_helpers import community, make_ctx, permit_articles, permit_claims, pitch_for
from newsroom.agents.explainer import ExplainerAgent, StoryDraft
from newsroom.langdetect import detect_language

ENGLISH = (
    "Winter tyres are required by law from 1 December, and drivers in Lapland must fit them "
    "before the first snow arrives."
)
FINNISH = (
    "Viikonloppu toi mukanaan poliisitehtäviä Rovaniemellä, ja kausityöntekijänä on hyvä "
    "ymmärtää paikalliset järjestyssäännöt."
)
SAMI = (
    "Oulu Pikisaari randageainnus lea dál vejolaš ieš válljet, mo čuvgehusat galget doaibmat. "
    "Dát lea oassin stuorát dutkanprošeavtta."
)
SAMI_NEWS = (
    "Dán vahku dieđut muitalit ahte biebmodoarjja lea šaddan dárbbašut, mátkkoštandivvagat "
    "eai boađe, ja Helsset lágida valáštallanruohttamiid."
)


def test_the_detector_tells_the_languages_apart():
    assert detect_language(ENGLISH) == "en"
    assert detect_language(FINNISH) == "fi"
    assert detect_language(SAMI) == "se"
    assert detect_language(SAMI_NEWS) == "se"  # real Sámi headlines that were filed as English
    assert detect_language("Det här är viktigt att veta för alla som arbetar i Lappland.") == "sv"
    assert detect_language("يجب على السائقين في لابلاند تركيب إطارات الشتاء قبل أول تساقط للثلج") == "ar"
    assert detect_language("Зимние шины обязательны по закону для водителей в Лапландии.") == "ru"


def test_the_detector_says_unsure_instead_of_guessing():
    assert detect_language("Hello there") is None
    assert detect_language("") is None
    assert detect_language(None) is None
    assert detect_language("Oulu Kela Migri 2026 12 24") is None


def _write(handler, community_id="oulu-international"):
    ctx = make_ctx()
    articles = permit_articles()
    pitch = pitch_for(community_id, articles)
    claims = permit_claims(pitch.id, articles)
    ctx.llm.register("explainer.write_story", handler)
    story = ExplainerAgent(ctx).write(pitch, claims, community(community_id), articles)
    return ctx, story


def test_a_draft_in_the_wrong_language_is_asked_for_again_in_english():
    answers = iter([SAMI, ENGLISH])
    prompts: list[str] = []

    def handler(payload):
        prompts.append(str(payload.get("feedback")))
        text = next(answers)
        return StoryDraft(headline="Winter tyres", standfirst=text, body_md=text)

    ctx, story = _write(handler)

    assert story.language == "en"
    assert len(prompts) == 2  # it was asked again
    assert any("another pass" in t.decision for t in ctx.store.traces())
    assert not [n for n in story.context_notes if "not English" in n.text]


def test_a_draft_that_stays_in_sami_is_filed_as_sami_and_flagged_not_as_english():
    def handler(_payload):
        return StoryDraft(headline="Čuvgehusat", standfirst=SAMI, body_md=SAMI + " " + SAMI_NEWS)

    ctx, story = _write(handler)

    assert story.language == "se"  # never labelled English
    notes = [n.text for n in story.context_notes if n.kind == "human_needed"]
    assert any("North Sámi" in text and "not English" in text for text in notes)
    assert any(t.step == "language" for t in ctx.store.traces())


def test_a_story_written_in_english_stays_english():
    def handler(_payload):
        return StoryDraft(headline="Winter tyres", standfirst=ENGLISH, body_md=ENGLISH)

    _, story = _write(handler)

    assert story.language == "en"
    assert not [n for n in story.context_notes if "not English" in n.text]


def test_machine_written_sami_is_never_published_but_english_and_finnish_are():
    from newsroom.agents.distributor import DistributorAgent
    from newsroom.domain import RiskTier

    def sami(_payload):
        return StoryDraft(headline="Čuvgehusat", standfirst=SAMI, body_md=SAMI + " " + SAMI_NEWS)

    ctx, sami_story = _write(sami, "sapmi-north")
    _, english_story = _write(
        lambda _p: StoryDraft(headline="Tyres", standfirst=ENGLISH, body_md=ENGLISH), "sapmi-north"
    )
    sapmi = community("sapmi-north")
    distributor = DistributorAgent(ctx)

    out_sami = distributor._publishable_languages(sami_story, [], sapmi, RiskTier.LOW)
    out_english = distributor._publishable_languages(english_story, [], sapmi, RiskTier.LOW)

    assert sami_story.language == "se" and [code for code, _ in out_sami] == []
    assert [code for code, _ in out_english] == ["en"]
    assert any(t.step == "language" and t.decision == "skipped" for t in ctx.store.traces())

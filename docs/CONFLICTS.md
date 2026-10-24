# Where the original plan conflicts with the challenge

The original project gist describes a **news intelligence platform**: ingest, investigate,
build a knowledge tree, deliver feeds. The Kaleva challenge asks for a **community medium**:
agents run the newsroom, a human is always in the loop, and the community is at the heart.
Most of the gist survives. These are the places where it does not, and what we did about each.

| # | Gist says | Challenge says | Resolution in this repo |
|---|-----------|----------------|-------------------------|
| 1 | Agents "autonomously decide what to investigate" and publish tailored feeds. | A human must always be in the loop. Responsible journalism only. | Agents **propose**; a LangGraph `interrupt` stops every story at a human gate. Nothing publishes without a named human approving the exact version, in every required role. The rule is enforced twice (graph and distributor) and tested. |
| 2 | Community Editor agent is "phase 6, after the first three agents". | The community is the heart, not an add-on. | The community desk, trusted-voice reviewers and community tips exist from the first prototype. They are first-class in the graph, not a late feature. |
| 3 | "Beyond a news aggregator": summarise and compare. | The medium must find, **select, produce and distribute** content. | Added the producing and distributing roles a real newsroom has: news editor, reporter, explainer, translator, copy chief, distributor, community desk. |
| 4 | Translate and summarise. | **Explanatory journalism**, not plain translation: what a decision means for someone new, with context from outside the story. | The explainer writes for a community profile and the Kaleva user-needs model. Context that cannot be sourced is marked `human_needed` instead of invented. |
| 5 | BBC News is the initial source; global event tree ("Global Affairs"). | Local: council decisions, official announcements, Kaleva. | Sources are Kaleva, Yle, BBC and official bodies. BBC and Yle are used as independent outside coverage to corroborate local stories. The tree is local-first (Oulu, Lapland, immigration, Sámi) with a world branch. |
| 6 | Treat the first source as the starting point for any story. | **Kaleva is paywalled.** We cannot read full articles. | Kaleva is `metadata_only`: headline, summary and link from its public feed, and we never get around the paywall. It still counts as an outlet that reported the event. Facts come from free full-text sources, official bodies, or licensed access if Kaleva grants it. This is a question for the organisers on Discord. |
| 7 | Store extracted article text; feeds with summaries. | Respect copyright and reuse rights. | Every source declares `reuse` (`link_and_summarize`, `licensed`, `official_public_info`). Default is summarise and link, never republish full text. |
| 8 | Languages: en, fi, sv. | Arabic, Ukrainian, Russian, Somali, North/Inari/Skolt Sámi, Swedish. | Language lanes have tiers. Tier C (Sámi) is **human-only**: the machine never produces publishable Sámi, it can only assist a human translator. Tier B/high-risk needs a language reviewer. |
| 9 | LLM confidence/uncertainty indicators. | Errors can affect residence permits, health, employment. | Risk tiers from policy data. High-risk stories need an official source linked, a subject expert, and back-translation checks. Confidence is never shown as a probability. |
| 10 | Source attribution and links. | **Trust**: why believe us over a Facebook group, and disclose AI openly. | Every story and message carries an AI disclosure, named approvers and links to originals. Corrections are published and say what was wrong. |
| 11 | Gemini API for all reasoning. | Community members may be refugees or other vulnerable people. | Community text is minimised before it reaches a hosted model (emails and numbers masked). Tips are leads, never evidence. Model layer is swappable if hosted inference is not acceptable. |
| 12 | Feeds in an API / web app. | Distribute **where communities already are**: messaging, audio, short groups. | Renditions per channel: messaging (length-limited), audio script, digest, web. The existing KalevaShorts web app is one channel, not the product. |
| 13 | One feed per user. | **Short life cycle**: a seasonal worker needs help on arrival, for a few months. | `Lifecycle` on each community and an onboarding pack for the first week. |
| 14 | One-way: tailored news for communities. | A **bridge** the other way: stories about communities for Finnish speakers. | `bridge_digest` builds a Finnish digest of stories about the communities. |
| 15 | Open source and self-hostable as a goal. | Business model: payer, willingness to pay, independence. | **Decided: we host it ourselves and it is not open source.** The business model is in `BUSINESS_MODEL.md`. Control of the service, of community data and of who can review is ours, which fits a medium that is accountable to its community. |
| 16 | Prompt injection: treat scraped text as untrusted. | (Compatible.) | Kept and tested: sanitising, delimiting, flagging, and never obeying instructions in articles. A flagged article cannot become a pitch. |

## Decisions we made so you do not have to start from nothing

* **First community:** international newcomers in Oulu (English first) and seasonal tourism workers
  in Lapland. They are the most numerous, have a clear urgent information need, and have payers
  (employers, tourism companies, universities). **Language pilot:** North Sámi, human-only, to show how
  small languages are handled without risking harm.
* **First language:** English, because machine translation quality allows meaningful review and the
  largest share of newcomers can read it. Ukrainian, Russian and Arabic follow with reviewers.
* **Model provider:** Gemini behind an interface, with an offline deterministic stand-in for tests and demos.

## Decisions made

1. **Official feeds enabled.** City of Oulu (news, and official notices and announcements, where council
   and board decisions are published), Oulu Region Public Transport, Mun Oulu, BusinessOulu, and three
   Migri feeds (press releases, customer bulletins, exceptions to service hours). Every feed was
   fetched and checked to return real RSS. The city's own channels all count as one outlet, so they
   cannot confirm each other.
2. **Trusted voices are the community's own admins and moderators.** An admin or moderator is appointed
   per community, and may review only in the roles and languages they were given. Their name and role
   are shown beside what they approved. **Editors search for people by name** and appoint community
   admins and moderators; a community admin appoints moderators in their own community only.
   Nobody in a community role can be the editor-in-chief.
3. **We host it, and it is not open source.** Documentation and the app no longer present it as either.
4. **A hosted model is acceptable for community messages**, with safeguards, and members are told plainly:
   a message containing an e-mail address or phone number goes straight to a person and the model never
   sees it; masking is a second safeguard; the model never receives names or account details; people,
   not the model, decide what is published or answered. These are tested (`tests/test_privacy.py`).

5. **Kaleva stays headline and link only** for now. It is confirmed as the current situation, and the
   code keeps it that way (`access: metadata_only`).
6. **The City of Oulu has no decisions feed beyond notices and announcements.** The official notices and
   announcements feed stays as the source for city decisions. If a decisions feed appears, add it to
   `config/sources.yaml`.

## Still open (need a human decision)

1. Who are the first real community admins and moderators in each community? (Editors can now search
   for them by name and appoint them.)
2. If Kaleva later grants licensed full-text access, change its `access` to `full`.

## Problems found while building, and how each was fixed

These were not in the brief. We found them by running the system and by testing it.

| Problem | What it would have caused | Fix |
|---|---|---|
| Kaleva full text is behind a paywall | Agents would have claimed facts from text they cannot read, or tried to get around the paywall | Kaleva is `metadata_only` (headline, summary, link). It counts as an outlet that reported the event. Claims that depend on unread text are listed as unknown. |
| A high-risk story with no official source was stuck: it had to link one, and nobody could add it | Stories ending as "blocked" and never reaching a human | A **source librarian** step attaches curated official references automatically (Migri, Kela, Tax Administration, occupational safety, InfoFinland and others). Editors check them and add more from the review desk. References are labelled as references, not proof. |
| A human's "request changes" was limited by the automatic revision limit | An editor's request silently ignored | Human requests have their own, separate limit. |
| A reviewer entered with an invalid name left the paused story unable to accept a valid review | A story stuck forever after one bad submission | Decisions are validated before the story is resumed. |
| Risk keywords matched inside other words (`jää` inside `työntekijää`) and missed Finnish word endings (`oleskeluluvan`) | A daycare story flagged high-risk; a permit story missed | Word-start matching and Finnish stems. |
| "Latest standards report" depended on database order | The wrong report used on MongoDB | Reports carry a timestamp. |
| Several feeds from one outlet looked like several outlets | Three Yle feeds counted as three independent confirmations | Outlets are normalised: all Yle feeds are one outlet, all Kaleva feeds one, all BBC one. |
| Gemini calls took about 80 seconds each at the default thinking level | A cycle taking many minutes | Thinking level `low` (about 27 seconds per call). A full live cycle still takes minutes, so production should run cycles in the background. |
| The offline model copies Finnish text into English stories and is not real translation | Misleading demos | Marked as an offline stand-in everywhere. Real quality needs Gemini plus reviewers. |
| Topic filing is keyword based | A hockey story filed under "Oulu region"; a bus timetable under "Economy" | Known limit. Editors approve restructuring, and a better matcher is the next step. |
| Every language held the whole story back until its reviewer approved | An editor could not publish a story in English because a Russian draft was waiting | A language reviewer gates only their language. The story publishes when the editor presses Publish, and each other language follows when approved. |
| Unresolved context ("a human must supply this") demanded an outside expert on every story, even low-risk ones | Almost nothing could be published with one click | It is now a warning shown to the editor. Outside experts are required only for high-risk stories, controlled by `review.high_risk_roles` in the policy. |
| `gemini-3.8-flash` took up to 149 seconds per call, and `gemini-2.5-flash` is refused for new keys (an earlier benchmark wrongly counted that fast 404 as success) | Cycles taking half an hour, or silently producing empty stories | Benchmarked again counting only valid parsed output: `gemini-3.1-flash-lite` 3.3 s, `gemini-3-flash-preview` 4.2 s, `gemini-3.8-flash` 38 s. Default is `gemini-3.1-flash-lite`. The service now tests the model at start-up and shows a warning if it is unusable. |
| Credentials pasted into chat (Gemini key, Atlas password, Clerk secret) | Exposed secrets | Rotate them. They are kept in git-ignored `.env` files only. |


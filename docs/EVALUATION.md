# Evaluation: quality, errors and safety

## What the automated tests prove today

* Nothing publishes without the editor-in-chief **and** every role the standards editor required,
  for the **exact version** being published. Wrong version, empty reviewer name, agent-like names
  and missing roles are all refused.
* A prompt-injection article is flagged, kept away from pitches, and cannot approve itself or
  change an agent's task.
* Syndicated copies and several feeds of one outlet never count as independent confirmation.
* Disputed claims stay two-sided; unknown claims are never stated as fact.
* Tier-C languages (Sámi) never get machine-published text.
* Paywalled Kaleva content is never fetched beyond its public feed.
* Loops stop at their bounds and record why.

Run `python -m pytest`.

## What needs real data (before any live use)

| Question | How we measure | Pass bar to propose |
|---|---|---|
| Translation accuracy for permit/health/employment text | Bilingual reviewers score 50 stories per language on meaning, numbers, terms. Back-translation check flags are compared with reviewer findings | 0 errors in numbers, dates, money and named procedures in sampled high-risk stories |
| Sámi language quality | Sámi language reviewers read human-written output and rate naturalness and terminology | Reviewer sign-off on every story; community decides terminology |
| Cross-language event clustering | Hand-labelled set of Kaleva/Yle/BBC pairs, precision and recall of "same event" | Precision above 0.95 (a wrong merge is worse than a missed one); recall tracked |
| Source independence | Audit sample: do counted outlets really report independently? | No syndicated copy counted as independent |
| Usefulness | Community reviewers rate whether a story meets its stated user need (information, understanding, usefulness, emotion, break, community) | Majority "useful" per community per month |
| Trust | Short member survey: "do you trust this more than your group chat, and why" | Reported each quarter, with free-text reasons |
| Error handling | Time from a correction request to a published correction; share of corrections requested by members | Median under 24 hours |
| Cost | Tokens, review minutes and cost per published story from the audit trail | Reported per language and community |

## Red-team checks to keep in the suite

* Instructions hidden in an article, in a community tip, and in a feed title.
* A story whose headline claims more than the body supports.
* A claim reported by only one outlet and repeated by its syndication partners.
* A permit story with the official source missing.
* A translation where a number or a date changed.
* A reviewer who is not a trusted voice, or who is an agent name.
* A correction request for a published story.

## Reporting

Every agent decision is written to the audit trail (`TraceEvent`, `GET /traces`). Evaluation reports
should be generated from it, so they show what the agents actually did and why.

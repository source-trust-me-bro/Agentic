# Following the Journalist's Guidelines

Source: Council for Mass Media in Finland, *Guidelines for Journalists*, in force 1 October 2024
(<https://jsn.fi/en/journalists-guidelines/>). They are self-regulation: the editorial office
decides, and a human is responsible. We map them by section so each is a checkable control,
not a promise. (Cite the numbered guideline text in review. This table uses section ranges.)

| Section | What it asks | How the newsroom does it |
|---|---|---|
| **Professional status (1–6)**: editorial decisions on journalistic grounds, independence, no benefits that compromise it, disclose conflicts involving the outlet or owners | Editors, not payers, decide. | Editorial decisions are made by named humans at the gate. Funders cannot choose stories or see drafts (see `BUSINESS_MODEL.md`). Funding that touches a story's subject is disclosed. |
| **Obtaining and publishing information (7–17)**: seek truthful information and check it, approach sources critically, acknowledge sources, separate fact, opinion and fiction, update as facts emerge, headlines match the story, no hidden advertising, disclose external funding | Truth, source criticism, transparency. | Investigator records evidence and independence; claims carry attribution and status. Standards editor blocks statements without a claim. Headline-vs-body check. Sources and links are mandatory. Context from outside the story must be sourced or marked `human_needed`. Syndicated copies are not corroboration. |
| **Interviewees and reply (18–22)**: right to reply when facing very negative coverage | Fairness. | Standards editor flags strongly negative claims about a named person or organisation and requires a reply attempt before approval is considered. |
| **Dignity, privacy, publicity (23–29)**: no stigma on ethnicity, health and similar; consent for sensitive matters; care with minors and the vulnerable | Respect, especially for vulnerable groups. | Dignity/privacy checks in standards review. Community text is minimised before it reaches a hosted model. Tips with private details go to a human, not into a story. |
| **Trials and crime (30–34)** | Care with names and guilt. | Out of scope for the prototype: crime and court stories are `needs_human` and not pitched automatically. |
| **Corrections (35–37)**: correct essential errors promptly and as widely as the original; say what was wrong and when fixed | Accountability. | Corrections are first-class (`Correction`), published by a named editor, stored with the story, and shown on every rendition. Community members can request one. |
| **Annex L1–L4: content by the public**: keep editorial and public content apart, moderate, let readers flag problems, remove incitement and privacy violations | Safe community participation. | Community desk moderates by rule before anything else. Tips are leads, never evidence. Members can flag stories and request corrections. |

## Human in the loop, concretely

1. The **editor-in-chief** must approve every story version.
2. **High-risk topics** (residence permits, health, employment, legal rights, safety) also need a
   **subject expert** and a linked official source.
3. **Machine-translated languages** that policy marks `required` need a **language reviewer**
   from the community. **Tier-C languages (Sámi) are written by a human.**
4. Reviewers must be named people. Community roles must come from the community's own trusted voices.
5. A new version after changes needs fresh approvals. Approving version 1 never publishes version 2.

## Disclosure

Every story and rendition says it was prepared by an AI-assisted newsroom and approved by a named
human editor, links the originals, and says how to report a mistake.

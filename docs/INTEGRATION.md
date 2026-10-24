# How the community is connected in the app

Three parts run together:

```
Browser ──► KalevaShorts web app (Next.js, Clerk sign-in)
                │  Bearer token (the signed-in user)
                ▼
            KalevaShorts API (FastAPI)  ── knows who you are and whether you are an editor
                │  X-Api-Key + a pseudonymous user reference (server to server only)
                ▼
            Agentic newsroom service (this repo) ── agents, human gate, audit trail
```

The web app talks only to the KalevaShorts API (one base URL, as before). That API relays
community and newsroom calls. Browsers never see the newsroom key.

## What a community member can do

| In the app | What happens |
|---|---|
| **Community** page, choose a community | The member is linked to a community and a language. The choice is stored under a pseudonymous reference, not a name or e-mail. |
| Read **Stories for you** | Only stories approved by named humans appear. Each shows the AI disclosure, who approved it, its sources (official references are labelled), context notes, and any published correction. |
| **Ask a question** | Answered only from published stories and maintained community notes. Otherwise it goes to an editor. Nothing is invented. |
| **Suggest a story** | Becomes a lead for the news editor. A tip is never published as fact. |
| **Share local knowledge** | Held for an editor, who can add it as a note credited to the community. |
| **Request a correction** | Opens a correction request on the chosen story. A named editor publishes the correction, and it says what was wrong. |
| **Your contributions** | The member sees the status of everything they sent. |

## What an editor does: press Publish

The agents do the routine work on their own: they collect the feeds on a schedule (every hour when a
real model is configured), group the same news from Kaleva, the City of Oulu and Migri, pick stories, check claims,
attach official references, write the explainer, draft the translations and run the standards check.
The editor finds finished stories waiting and presses **Publish**.

* **Ordinary stories need only the editor.** One click publishes the English story.
* **Other languages go out when their own reviewer approves them.** A language reviewer (a community
  moderator with that language) never holds the English story back. When they approve, the story is
  republished with that language added, and their name is shown beside it.
* **High-risk stories** (permits, health, employment, legal rights, safety) need extra named reviewers
  by default, because a mistake there harms people. The button then says "Approve" and lists what else
  is needed. To let the editor-in-chief alone approve them, set `review.high_risk_roles: []` in
  `config/policy.yaml`. That is a deliberate choice, and the editor still decides.
* **The agents never approve.** An approval always carries a named person.

Details of the desk:

On the **Editor** page, the **Newsroom desk**:

1. Runs a newsroom cycle (collect, cluster the same news from Kaleva, the City of Oulu and Migri into events, pick stories).
2. Sees each prepared story with its risk level, the sources the agents attached (marked "Added by the newsroom"), translation status and what is still needed.
3. **Checks the sources and adds more** if needed.
4. Approves, requests changes, or rejects. The reviewer is the signed-in editor. The name comes from the account, not from a form field.

## Trusted voices: the community's admins and moderators

The community's own admins and moderators are its trusted voices.

* They are **appointed per community**. **Editors search for the person by name** and appoint community
  admins and moderators in any community. A community admin can appoint moderators in their own community,
  but not other admins. The form is on the **Editor** page (editors) and on the **Moderate** page
  (community admins). Search shows the name and a masked e-mail hint (`a***@example.com`) so two people
  with the same name can be told apart; the full address is never shown. The person needs to have
  signed in once to be found.
* An appointment names the **review roles** the person may hold (community reviewer, language reviewer,
  subject expert) and the **languages** they may approve. A language reviewer approves only their own languages.
* Moderators open **Moderate** in the sidebar. They see only stories for their own community and can act
  only in the roles they were given. They cannot be the editor-in-chief and cannot add sources.
* Their name and role (for example "Aino, community moderator") are shown to readers beside what they approved.
* A story cannot publish until every required role has approved by a person who holds that role.

## Identity and privacy

* The newsroom never receives an account id, name or e-mail. It receives `sha256("kalevashorts:" + account id)`.
* A hosted model (Google Gemini) helps editors read and sort contributions. A message containing an
  e-mail address or phone number goes straight to a person and the model never sees it. Masking is a
  second safeguard. The model never receives names or account details. The form tells members this plainly.
* Reviewer names for approvals are taken from the authenticated account on the server.

## Running everything

```powershell
# 1. Newsroom service (this repo), port 8100
cd D:\Agentic
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m newsroom serve

# 2. KalevaShorts API, port 8000  (needs NEWSROOM_URL and NEWSROOM_API_KEY in its .env)
cd D:\Backend
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# 3. Web app, port 3000
cd D:\Frontend
pnpm dev
```

`NEWSROOM_API_KEY` in the backend `.env` must equal `API_KEY` in the newsroom `.env`.

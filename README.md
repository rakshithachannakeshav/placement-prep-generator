# Placement Prep Report Generator

A multi-agent LLM system that generates a company + role prep dossier
from a resume and (optional) job description — grounded in live web
research and self-fact-checked before it's returned. If you leave the
role blank, it falls back to a general company-fit report with topics
to prepare, instead of assuming a specific role.

**Live**: https://placement-prep-generator.onrender.com
(free tier — spins down after inactivity, first request after idle
takes ~30-50s to wake up)

**100% free-tier**: Groq for the LLM, DuckDuckGo (`ddgs`) for search,
TF-IDF for retrieval. No paid API, no model downloads.

## Architecture

```
        ┌─> research_company  (live web search on the company) ─┐
START ──┤                                                        ├─> draft ─> fact_check ─┬─> draft (revise)
        └─> research_role_fit (RAG over resume + JD)      ───────┘                        └─> finalize ─> END
```

- **research_company**: searches the web (DuckDuckGo, no key needed) for
  the company's interview process, culture, and recent focus areas.
  Filters results to those actually mentioning the company (short names
  like "Q2" otherwise pull in unrelated noise), and degrades to "no
  results" rather than failing the request if every search backend is
  unavailable.
- **research_role_fit**: retrieves the resume/JD chunks most relevant to
  the target role using TF-IDF + cosine similarity (`app/rag.py`) — or,
  if no role was given, the chunks most relevant to the candidate's
  strongest general skills.
- **draft**: a Groq LLM call synthesizes both into a dossier. With a
  role, that's a role-specific 5-section report; without one, it's a
  6-section report that suggests roles/teams to target and lists
  concrete topics to prepare.
- **fact_check**: a second (smaller/faster) Groq call checks the draft
  against the research context for unsupported claims. If it finds any,
  the draft node re-runs with that feedback — up to `MAX_REVISIONS`
  times (self-correcting revision loop).
- **finalize**: returns the last clean (or max-revised) draft.

Both Groq calls retry on rate limits (using the wait time Groq's own
error reports) and fall back to a secondary model once if the
configured model has been decommissioned — Groq's hosted model lineup
changes over time.

The frontend (`backend/static/index.html`) is a single static page with
no build step, served by FastAPI itself at `/`. The report endpoint
streams progress over Server-Sent Events, so the page shows live status
("Researching company...", "Drafting...", "Fact-checking...",
"Revising draft (attempt N)...") instead of a blank wait.

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your free Groq key from https://console.groq.com/keys
uvicorn app.main:app --reload --port 8001
```

Then open `http://localhost:8001/` for the form, or hit the API directly:

```bash
curl -N -X POST http://localhost:8001/generate-report \
  -F "company=Google" \
  -F "role=" \
  -F "jd_text=" \
  -F "resume=@/path/to/your/resume.pdf"
```
(role can be left blank for the general company-fit mode; the response
is a Server-Sent Events stream — `-N` disables curl's output buffering
so you see status events as they arrive)

## Deployment

`render.yaml` at the repo root is a Render Blueprint: connect the repo
at [render.com](https://render.com) → New → Blueprint, set
`GROQ_API_KEY` in the dashboard (the only secret, never committed), and
it deploys `backend/` with `DEBUG=false`. Pushes to `main` auto-redeploy.

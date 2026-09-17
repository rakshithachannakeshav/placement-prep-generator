# Placement Prep Report Generator

A multi-agent LLM system that generates a company + role prep dossier
from a resume and (optional) job description — grounded in live web
research and self-fact-checked before it's returned.

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
- **research_role_fit**: retrieves the resume/JD chunks most relevant to
  the target role using TF-IDF + cosine similarity (`app/rag.py`).
- **draft**: Groq LLM synthesizes both into a 5-section dossier.
- **fact_check**: a second Groq call checks the draft against the
  research context for unsupported claims. If it finds any, the draft
  node re-runs with that feedback — up to `MAX_REVISIONS` times
  (self-correcting revision loop).
- **finalize**: returns the last clean (or max-revised) draft.

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your free Groq key from https://console.groq.com/keys
uvicorn app.main:app --reload --port 8001
```

(Port 8001, not 8000 — same fix you already made for CareHub.)

## Try it

```bash
curl -X POST http://localhost:8001/generate-report \
  -F "company=Groq" \
  -F "role=Software Engineer Intern" \
  -F "jd_text=Looking for engineers with Python, ML systems, and API experience." \
  -F "resume=@/path/to/your/resume.pdf"
```

## What's tested vs. what needs your key

Everything except the actual Groq/DuckDuckGo network calls has been
tested end-to-end in this build (PDF ingestion on your real resume, the
TF-IDF retriever, and the full graph wiring including the revision loop
— with the LLM/search calls mocked). Once you drop in your free Groq
key, the `draft` and `fact_check` nodes go live; DuckDuckGo search needs
no key at all.

## Next steps (in rough priority order)

1. Get it running locally with your Groq key, sanity-check one real report.
2. Add a minimal frontend (a single form + result view is enough — doesn't
   need to be the full Vite/React app yet).
3. Deploy the backend somewhere free (Render/Railway free tier) so you
   have a live link, not just local — that's what actually proves
   "hands-on deployment" in an interview.
4. Once it's live, update your resume bullet with a real number: how
   many reports you've generated for yourself/friends, or how long one
   takes vs. manual research.

## Resume bullet (draft, once you've run it a few times)

**Placement Prep Report Generator** | Python, LangGraph, Groq, FastAPI, RAG
- Architected a multi-agent LLM system (Manager-of-Managers pattern) with
  parallel Research agents, an LLM Drafting agent, and a self-correcting
  Fact-Check agent that revises the draft against retrieved evidence.
- Built a TF-IDF retrieval pipeline over resume/JD text and live web
  search results to ground every generated claim in real sources.
- Deployed as a FastAPI service on a fully free-tier LLM stack (Groq),
  generating [X] personalized prep dossiers in [Y seconds/minutes] each.

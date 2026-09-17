"""
Manager-of-Managers style multi-agent graph for the Placement Prep Report
Generator, built on LangGraph.

Flow:
    START ─┬─> research_company ─┐
           └─> research_role_fit ┴─> draft ─> fact_check ─┬─> draft (revise, if issues found)
                                                           └─> finalize ─> END

- research_company / research_role_fit run in parallel (fan-out from START,
  joined automatically by LangGraph before `draft` runs).
- fact_check re-reads the draft against the research context and can send
  it back to `draft` with concrete feedback, up to MAX_REVISIONS times
  (the self-correcting revision loop).
"""
import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .config import MAX_REVISIONS, GROQ_FACT_CHECK_MODEL
from .llm import chat
from .rag import TfidfRetriever
from .web_search import search_as_context


class ReportState(TypedDict):
    company: str
    role: str
    jd_text: str
    resume_text: str
    company_research: str
    role_fit_research: str
    draft: str
    issues: list[str]
    revision_count: int
    final_report: str


def _build_retriever(state: ReportState) -> TfidfRetriever:
    retriever = TfidfRetriever()
    retriever.build({"resume": state["resume_text"], "jd": state["jd_text"]})
    return retriever


def research_company_node(state: ReportState) -> dict:
    company = state["company"]
    query = f'"{company}" interview process technical rounds culture 2026'
    context = search_as_context(query, max_results=5, prefer_term=company)
    return {"company_research": context}


def research_role_fit_node(state: ReportState) -> dict:
    retriever = _build_retriever(state)
    role = state["role"].strip()
    query = f"skills relevant to {role}" if role else "key technical skills and strongest projects"
    hits = retriever.query(query, k=5)
    joined = "\n".join(f"[{h.source}] {h.text}" for h in hits)
    return {"role_fit_research": joined}


def draft_node(state: ReportState) -> dict:
    feedback = ""
    if state.get("issues"):
        feedback = (
            "\n\nThe previous draft had these issues — fix them in this revision:\n"
            + "\n".join(f"- {i}" for i in state["issues"])
        )

    role = state["role"].strip()

    if role:
        role_line = f"for the role of {role}"
        sections = """1. Company Snapshot — what they do, recent focus areas, interview format if known
2. Likely Technical Focus Areas — grounded ONLY in the research above, not guesses
3. Candidate Fit — 3-4 bullets connecting the candidate's actual background to this role
4. Gaps to Prepare — honest gaps between the candidate's background and likely expectations
5. Sample Interview Questions — 5-6 tailored questions"""
    else:
        role_line = "who hasn't picked a specific role yet"
        sections = """1. Company Snapshot — what they do, recent focus areas, interview format if known
2. Key Areas the Company Likely Values — grounded ONLY in the research above, not guesses
3. How Your Background Fits — 3-4 bullets connecting the candidate's actual background to this company generally
4. Suggested Roles/Teams to Target — 2-3 roles that plausibly fit the candidate's background, clearly labeled as suggestions to confirm on the company's careers page, not confirmed openings
5. Topics to Prepare — the concrete technical/domain topics the candidate should study before interviewing here
6. Sample Interview Questions — 5-6 general questions likely for someone with this background at this company"""

    prompt = f"""You are writing a placement-prep dossier for a student interviewing at {state['company']}
{role_line}.

Job description (if provided):
{state['jd_text'] or '(not provided)'}

Company research (from live web search):
{state['company_research']}

Candidate's relevant background (retrieved from their resume):
{state['role_fit_research']}
{feedback}

Write a concise, well-structured prep dossier with these sections:
{sections}

Only state facts that are supported by the research/context above. If something is unknown, say so
rather than inventing it."""

    draft = chat(prompt, system="You are a precise, evidence-grounded career-prep writing assistant.")
    return {"draft": draft, "issues": []}


def fact_check_node(state: ReportState) -> dict:
    prompt = f"""Check the DRAFT below against the SOURCE CONTEXT. Flag any claim in the draft that is
NOT supported by the source context (e.g. invented statistics, made-up interview formats, unverifiable
claims about the company). Respond with ONLY a JSON object: {{"issues": ["...", "..."]}}.
If there are no unsupported claims, respond with {{"issues": []}}.

SOURCE CONTEXT:
{state['company_research']}
{state['role_fit_research']}

DRAFT:
{state['draft']}"""

    raw = chat(
        prompt,
        system="You are a strict fact-checker. Output valid JSON only, nothing else.",
        model=GROQ_FACT_CHECK_MODEL,
    )
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        issues = json.loads(cleaned).get("issues", [])
    except (json.JSONDecodeError, AttributeError):
        issues = []

    return {"issues": issues, "revision_count": state.get("revision_count", 0) + 1}


def route_after_fact_check(state: ReportState) -> str:
    if state["issues"] and state["revision_count"] < MAX_REVISIONS:
        return "draft"
    return "finalize"


def finalize_node(state: ReportState) -> dict:
    return {"final_report": state["draft"]}


def build_graph():
    graph = StateGraph(ReportState)

    graph.add_node("research_company", research_company_node)
    graph.add_node("research_role_fit", research_role_fit_node)
    graph.add_node("draft", draft_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "research_company")
    graph.add_edge(START, "research_role_fit")
    graph.add_edge("research_company", "draft")
    graph.add_edge("research_role_fit", "draft")
    graph.add_edge("draft", "fact_check")
    graph.add_conditional_edges("fact_check", route_after_fact_check, {"draft": "draft", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()

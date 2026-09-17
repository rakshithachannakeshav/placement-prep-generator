import json
import logging
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import DEBUG
from .graph import build_graph
from .ingestion import load_pdf

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Placement Prep Report Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


_graph = build_graph()


class ReportResponse(BaseModel):
    final_report: str
    revision_count: int
    company_research: str
    role_fit_research: str


STATUS_LABELS = {
    "research_company": "Researching company...",
    "research_role_fit": "Analyzing resume fit...",
    "fact_check": "Fact-checking draft...",
    "finalize": "Finalizing report...",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/generate-report")
async def generate_report(
    company: str = Form(...),
    role: str = Form(""),
    jd_text: str = Form(""),
    resume: UploadFile = File(...),
):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await resume.read())
        tmp_path = Path(tmp.name)

    try:
        resume_text = load_pdf(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    initial_state = {
        "company": company,
        "role": role,
        "jd_text": jd_text,
        "resume_text": resume_text,
        "issues": [],
        "revision_count": 0,
    }

    def event_stream():
        state = dict(initial_state)
        draft_runs = 0
        try:
            for step in _graph.stream(initial_state):
                for node_name, node_output in step.items():
                    state.update(node_output)

                    if node_name == "draft":
                        draft_runs += 1
                        label = "Drafting report..." if draft_runs == 1 else f"Revising draft (attempt {draft_runs})..."
                    else:
                        label = STATUS_LABELS.get(node_name, node_name)

                    yield _sse("status", {"node": node_name, "label": label})

            result = ReportResponse(
                final_report=state["final_report"],
                revision_count=state["revision_count"],
                company_research=state["company_research"],
                role_fit_research=state["role_fit_research"],
            )
            yield _sse("result", result.model_dump())
        except Exception as exc:
            logger.exception("Error during report generation stream")
            payload = {"error": type(exc).__name__, "message": str(exc)}
            if DEBUG:
                payload["traceback"] = traceback.format_exc()
            yield _sse("error", payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mounted last so it doesn't shadow the API routes above; serves static/index.html at "/".
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

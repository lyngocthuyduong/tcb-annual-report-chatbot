"""HTTP API + minimal UI. Single-process, one answer at a time."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import chat, conf

UI = Path(__file__).with_name("ui.html")
PDF = Path(conf.PDF_PATH)
HTTP = {"timeout": 504, "unavailable": 503, "rejected": 502, "config": 500, "generation": 502, "retrieval": 500, "llm": 502}


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


def create_app(engine, trace_dir: Path | None = None, pdf: Path = PDF) -> FastAPI:
    app = FastAPI(title="TCB Annual Report 2025 chatbot", docs_url="/docs")
    sessions: dict[str, chat.Session] = {}
    lock = threading.Lock()
    tdir = Path(trace_dir) if trace_dir else None

    @app.get("/health")
    def health():
        m = engine.rt.ix.manifest
        return {"status": "ok", "provider": engine.prov.name, "model": engine.prov.model,
                "index": {"nodes": m["counts"]["nodes"], "embedding": m["embedding"]["model"],
                          "built_at": m["built_at"]}}

    @app.post("/chat")
    def chat_(body: ChatIn):
        sid = body.session_id or uuid.uuid4().hex[:12]
        with lock:
            s = sessions.setdefault(sid, chat.Session())
            if len(sessions) > 200:
                sessions.pop(next(iter(sessions)))
            a = engine.ask(body.question, s.turns)
            if a.status != "error":
                s.add(chat.Turn(body.question, a.standalone, a.answer, a.status))
        if tdir:
            try:
                tdir.mkdir(parents=True, exist_ok=True)
                with open(tdir / "chat.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"session_id": sid, **a.to_dict(trace=True)}, ensure_ascii=False) + "\n")
            except OSError:
                pass                                           # tracing must never break an answer
        d = {"session_id": sid, **a.to_dict()}
        return JSONResponse(d, status_code=HTTP.get(a.error_kind or "llm", 502) if a.status == "error" else 200)

    @app.get("/evidence/{chunk_id}")
    def evidence(chunk_id: str):
        n = engine.rt.ix.nodes.get(chunk_id)
        if not n:
            raise HTTPException(404, "unknown chunk_id")
        return {k: n.get(k) for k in ("chunk_id", "text", "printed_page_number", "pdf_page_index", "side", "section",
                                      "content_type", "table_id", "form_code", "unit_header", "page_number_source")}

    @app.get("/pdf")
    def pdf_():
        if not pdf.exists():
            raise HTTPException(404, "data/ar.pdf not mounted")
        return FileResponse(pdf, media_type="application/pdf")

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return UI.read_text(encoding="utf-8")

    return app

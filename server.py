"""
server.py
─────────
FastAPI server for invoice classification via COA agent queue.

Endpoints:
  POST /upload/coa            — Upload & parse COA PDF → returns tree + coa_id
  POST /process               — Upload invoices + reference coa_id → triggers parallel processing
  GET  /status/{batch_id}     — SSE stream; emits result events as each invoice completes
  GET  /health                — Health check
"""

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Your existing imports ─────────────────────────────────────────────────────
from utils.logger import configure_logging
from config import DEBUG
from mistral_comp.invoice_parser import InvoiceParser
from agent.agent_qeue import get_agent_queue
from coa_utils.main import parse_coa

# ── Setup ─────────────────────────────────────────────────────────────────────
configure_logging(debug=DEBUG)
log = logging.getLogger(__name__)

TMP_ROOT = Path("/tmp")   # all uploads live under /tmp/<batch_id>/

app = FastAPI(title="Invoice COA Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (replace with Redis / DB for production)
_coa_store: Dict[str, Dict[str, Any]] = {}      # coa_id → {tree, path}
_batch_results: Dict[str, Dict] = {}             # batch_id → {status, results: {task_id: ...}}
_batch_events: Dict[str, asyncio.Queue] = {}     # batch_id → asyncio.Queue of SSE payloads

invoice_parser = InvoiceParser()


# ── Models ────────────────────────────────────────────────────────────────────
class COAUploadResponse(BaseModel):
    coa_id: str
    tree: Dict[str, Any]
    message: str


class ProcessResponse(BaseModel):
    batch_id: str
    task_ids: List[str]
    invoice_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────
async def _save_upload(file: UploadFile, dest: Path) -> None:
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 256):  # 256 KB chunks
            await f.write(chunk)


def _build_coa_tree(pdf_path: str) -> Dict[str, Any]:
    coa_tree = parse_coa(pdf_path)
    raw = coa_tree.model_dump(mode="json")
    return raw.get("hierarchy", {})


def _delete_dir(path: Path) -> None:
    """Remove a directory tree, silently ignoring errors."""
    try:
        shutil.rmtree(path, ignore_errors=True)
        log.info("Deleted temp dir: %s", path)
    except Exception as exc:
        log.warning("Could not delete %s: %s", path, exc)


async def _process_invoice(
    *,
    pdf_path: str,
    file_name: str,
    expenses_tree: Dict[str, Any],
    batch_id: str,
    task_id: str,
) -> None:
    """Parse one invoice then enqueue it; put result on the batch event queue."""
    queue = _batch_events.get(batch_id)
    agent_queue = get_agent_queue()

    try:
        # Emit "started" event
        if queue:
            await queue.put({
                "event": "invoice_started",
                "task_id": task_id,
                "file_name": file_name,
            })

        success, invoice_data, error = await invoice_parser.parse_invoice(pdf_path=pdf_path)

        if not success:
            raise ValueError(f"Parsing failed: {error}")

        invoice = invoice_data.model_dump()
        line_items = invoice["line_items"]
        vendor = invoice_data.header.vendor_name

        # Emit line items found
        if queue:
            await queue.put({
                "event": "invoice_parsed",
                "task_id": task_id,
                "file_name": file_name,
                "vendor": vendor,
                "line_item_count": len(line_items),
            })

        # Concatenate descriptions (matches your existing main.py logic)
        total = "\n".join(li["description"] for li in line_items)

        enqueued_id = await agent_queue.enqueue_request(
            task_id=task_id,
            batch_id=batch_id,
            line_item=total,
            file_name=file_name,
            expenses_tree=expenses_tree,
        )

        response = await agent_queue.wait_for_response(enqueued_id, timeout=600)

        result = {
            "task_id": task_id,
            "file_name": file_name,
            "vendor": vendor,
            "line_item_count": len(line_items),
            "selected_leaf": response.selected_leaf,
            "status": "done",
        }

        _batch_results[batch_id]["results"][task_id] = result

        if queue:
            await queue.put({"event": "invoice_done", **result})

    except Exception as exc:
        log.exception("Invoice processing error for task %s", task_id)
        err_result = {
            "task_id": task_id,
            "file_name": file_name,
            "status": "error",
            "error": str(exc),
        }
        if batch_id in _batch_results:
            _batch_results[batch_id]["results"][task_id] = err_result
        if queue:
            await queue.put({"event": "invoice_error", **err_result})

    finally:
        # Check if all tasks in the batch are done
        if batch_id in _batch_results:
            results = _batch_results[batch_id]["results"]
            total_tasks = _batch_results[batch_id]["total"]
            finished = sum(1 for r in results.values() if r.get("status") in ("done", "error"))
            if finished >= total_tasks:
                _batch_results[batch_id]["status"] = "complete"
                # ── Delete the batch upload folder now that everything is processed ──
                batch_dir = TMP_ROOT / batch_id
                await asyncio.get_event_loop().run_in_executor(None, _delete_dir, batch_dir)
                if queue:
                    await queue.put({"event": "batch_complete", "batch_id": batch_id})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload/coa", response_model=COAUploadResponse)
async def upload_coa(file: UploadFile = File(...)):
    """Upload a COA PDF, parse it, store the tree, return coa_id + tree."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "COA file must be a PDF.")

    coa_id = uuid.uuid4().hex
    coa_dir = TMP_ROOT / f"coa_{coa_id}"
    coa_dir.mkdir(parents=True, exist_ok=True)
    dest = coa_dir / "coa.pdf"
    await _save_upload(file, dest)

    try:
        tree = await asyncio.get_event_loop().run_in_executor(
            None, _build_coa_tree, str(dest)
        )
    except Exception as exc:
        await asyncio.get_event_loop().run_in_executor(None, _delete_dir, coa_dir)
        raise HTTPException(500, f"COA parsing failed: {exc}")
    finally:
        # Always clean up the COA file — tree is now in memory
        await asyncio.get_event_loop().run_in_executor(None, _delete_dir, coa_dir)

    _coa_store[coa_id] = {"tree": tree}
    log.info("COA uploaded: coa_id=%s keys=%s", coa_id, list(tree.keys())[:5])

    return COAUploadResponse(
        coa_id=coa_id,
        tree=tree,
        message=f"COA parsed successfully. Found {len(tree)} top-level accounts.",
    )


@app.post("/process", response_model=ProcessResponse)
async def process_invoices(
    coa_id: str = Form(...),
    invoices: List[UploadFile] = File(...),
):
    """Kick off parallel processing of up to 5 invoices against a parsed COA."""
    if coa_id not in _coa_store:
        raise HTTPException(404, f"coa_id '{coa_id}' not found. Upload COA first.")

    if len(invoices) > 50:
        raise HTTPException(400, "Maximum 50 invoices per batch.")

    for f in invoices:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"'{f.filename}' is not a PDF.")

    expenses_tree = _coa_store[coa_id]["tree"]
    batch_id = uuid.uuid4().hex
    task_ids: List[str] = []

    # ── Create /tmp/{batch_id}/ and save all invoices there ──────────────────
    batch_dir = TMP_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved: List[tuple] = []
    for invoice_file in invoices:
        task_id = uuid.uuid4().hex
        dest = batch_dir / f"{task_id}_{invoice_file.filename}"
        await _save_upload(invoice_file, dest)
        saved.append((task_id, str(dest), invoice_file.filename))
        task_ids.append(task_id)

    _batch_results[batch_id] = {
        "status": "running",
        "total": len(task_ids),
        "results": {},
    }
    _batch_events[batch_id] = asyncio.Queue()

    # Start agent workers once (idempotent)
    agent_queue = get_agent_queue()
    agent_queue.start_workers(3)

    # Fire-and-forget parallel tasks
    for task_id, pdf_path, file_name in saved:
        asyncio.create_task(
            _process_invoice(
                pdf_path=pdf_path,
                file_name=file_name,
                expenses_tree=expenses_tree,
                batch_id=batch_id,
                task_id=task_id,
            )
        )

    log.info("Batch %s started with %d invoice(s)", batch_id, len(task_ids))
    return ProcessResponse(batch_id=batch_id, task_ids=task_ids, invoice_count=len(task_ids))


@app.get("/status/{batch_id}")
async def batch_status_stream(batch_id: str):
    """
    SSE endpoint. Streams JSON events until the batch is complete.
    Event shape: data: {"event": "<type>", ...}\n\n
    """
    if batch_id not in _batch_events:
        raise HTTPException(404, f"batch_id '{batch_id}' not found.")

    queue = _batch_events[batch_id]

    async def event_generator():
        # Immediately send any already-complete results
        existing = _batch_results.get(batch_id, {}).get("results", {})
        for result in existing.values():
            yield f"data: {json.dumps({'event': 'invoice_done', **result})}\n\n"

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30)
                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get("event") == "batch_complete":
                    break
            except asyncio.TimeoutError:
                yield "data: {\"event\": \"ping\"}\n\n"  # keep-alive

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Prompt management ─────────────────────────────────────────────────────────
PROMPTS_DIR = Path("prompts")

PROMPT_FILES = {
    "dr_system_prompt": PROMPTS_DIR / "dr_system_prompt.txt",
}


@app.get("/prompts")
async def get_prompts():
    """Return all editable prompt files as {key: content}."""
    result = {}
    for key, path in PROMPT_FILES.items():
        try:
            result[key] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            result[key] = ""
    return result


@app.post("/prompts")
async def save_prompts(payload: Dict[str, str]):
    """
    Persist updated prompt content to disk.
    Body: {key: new_content, ...}
    Only keys listed in PROMPT_FILES are accepted.
    """
    saved = []
    for key, content in payload.items():
        if key not in PROMPT_FILES:
            raise HTTPException(400, f"Unknown prompt key: '{key}'")
        path = PROMPT_FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log.info("Prompt '%s' updated (%d chars)", key, len(content))
        saved.append(key)
    return {"saved": saved, "message": f"Saved {len(saved)} prompt(s) successfully."}


# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("frontend.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Frontend not found. Place frontend.html alongside server.py.</h1>")
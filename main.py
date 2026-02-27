"""
main.py
───────
Entry point for invoice classification via the COA agent queue.

Flow:
  1. Parse the invoice PDF → extract line items
  2. Start background agent workers
  3. Enqueue each line item as a separate task
  4. Wait for all results
  5. Print / return the selected ledger per line item
"""

import asyncio
import uuid

from utils.logger import configure_logging
from config import DEBUG
from mistral_comp.invoice_parser import InvoiceParser
from agent.agent_qeue import get_agent_queue
from agent.expense_tree import EXPENSES_TREE
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH = "/home/soham/Documents/orbtl/Hypro-2/input/45.pdf"
N_WORKERS = 3   # concurrent agent workers (each does multi-turn Bedrock calls)

configure_logging(debug=DEBUG)

invoice_parser = InvoiceParser()


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():

    # 1. Start background workers FIRST (before any awaits that take time)
    agent_queue = get_agent_queue()
    agent_queue.start_workers(N_WORKERS)

    p = Path(PDF_PATH)
    
    file_name = p.name

    # 2. Parse the invoice
    success, invoice_data, error = await invoice_parser.parse_invoice(pdf_path=PDF_PATH)

    if not success:
        print(f"❌ Invoice parsing failed: {error}")
        return

    invoice    = invoice_data.model_dump()
    vendor     = invoice_data.header.vendor_name
    line_items = invoice["line_items"]
    batch_id   = uuid.uuid4().hex
    task_id = uuid.uuid4().hex

    print(f"📄 Invoice parsed — {len(line_items)} line item(s), vendor: {vendor}")
    
    total = ""
    for l in line_items:
        total += l["description"] + "\n"
        

    # 3. Enqueue every line item as its own task
    #    Each gets a unique task_id; they all share the same batch_id
    task_id = await agent_queue.enqueue_request(
        task_id=task_id,
        batch_id=batch_id,
        line_item=total,
        file_name=file_name,
        expenses_tree=EXPENSES_TREE,
    )

    print(f"📦 Enqueued {task_id} task(s) under batch {batch_id}")

    # 4. Wait for all results
    response = await agent_queue.wait_for_response(task_id, timeout=600)

    # 5. Print results aligned with original line items
    print("\n── Results ──────────────────────────────────────────────────")
    print(response.selected_leaf)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
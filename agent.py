"""
agent.py
────────
Main entry point for the COA invoice classification agent.

Key functions:
  run_agent(invoice_description, vendor_name)  → selected ledger name (str)
  classify_batch(invoices)                      → dict of id → ledger name

Each run:
  - Creates a fresh AgentMemory
  - Opens a Langfuse v3 trace (root span) for the invoice
  - Runs the Bedrock converse() agentic loop
  - Every LLM call → nested "generation" span
  - Every tool call → nested "span"
  - Terminates when select_leaf is called or MAX_TURNS is exceeded
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError
from langfuse import get_client

from config import EXPENSES_TREE, MODEL_ID, bedrock_client
from memory import AgentMemory, initialize_memory, is_done
from tools import TOOL_CONFIG, execute_tool

# ── Safety cap ────────────────────────────────────────────────────────────────
MAX_TURNS = 30  # max Bedrock converse() calls per invoice

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
You are a Chart of Accounts (COA) classification agent. Your ONLY job is to read \
an invoice description and navigate the Expenses ledger tree to find the single \
most appropriate leaf node (ledger account) to assign the invoice to.

Invoice description: {invoice_description}
Vendor (if known): {vendor_name}

━━━ TREE STRUCTURE ━━━
The tree is a hierarchy. Nodes are either:
  - FOLDER  → contains more nodes inside (type: "folder")
  - LEAF    → a final ledger account with no children (type: "leaf")
Your answer must always be a LEAF node.

━━━ YOUR TOOLS ━━━
  get_children(path)           → See immediate children of a node + their states
  navigate_to(path)            → Move into or back to any seen path
  update_node_states(updates)  → Mark nodes DISCARDED (skip by name) or EXHAUSTED (explored, nothing)
  get_leaf_nodes(path)         → Get all leaves under a path — use when you're confident of the subtree
  get_unexplored_paths()       → See what's left if you feel stuck or lost
  select_leaf(path, leaf_name) → Your final answer — only call when certain

━━━ NODE STATES ━━━
  UNEXPLORED  → Not yet visited. You should explore these.
  IN_PROGRESS → Currently being explored (where you are now).
  EXHAUSTED   → Explored fully, nothing relevant found. Do NOT re-enter.
  DISCARDED   → Rejected by name — clearly irrelevant. Do NOT enter.
  SELECTED    → Final answer (only one leaf).

━━━ STRICT RULES ━━━
1. Always call get_children before entering any new node for the first time.
2. When you are confident you are in the right subtree, call get_leaf_nodes — \
   if there are ≤12 leaves, pick directly and call select_leaf.
3. When you realize you went the wrong way, call navigate_to with the correct \
   backtrack path — do NOT try to re-enter EXHAUSTED nodes.
4. Discard obviously irrelevant nodes immediately using update_node_states with \
   state=DISCARDED, so you don't waste turns on them.
5. If you are unsure what remains to explore, call get_unexplored_paths first.
6. select_leaf is your only terminal action — call it exactly once, when certain.
7. Never enter a node marked EXHAUSTED or DISCARDED.

━━━ STRATEGY ━━━
Start at ["Expenses"]. Call get_children(["Expenses"]) to see the top-level options. \
Discard clearly irrelevant branches immediately. Enter the most promising branch \
and repeat. When you reach a small enough set of leaves (≤12), use get_leaf_nodes \
and pick the best match.
"""


def _build_system_prompt(invoice_description: str, vendor_name: Optional[str]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        invoice_description=invoice_description,
        vendor_name=vendor_name or "Unknown",
    )


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent(
    invoice_description: str,
    vendor_name: Optional[str] = None,
) -> str:
    """
    Classify a single invoice. Returns the selected ledger name.
    Raises RuntimeError if classification fails or MAX_TURNS exceeded.
    """
    langfuse = get_client()
    memory   = initialize_memory(EXPENSES_TREE, invoice_description, vendor_name)

    with langfuse.start_as_current_observation(
        as_type="span",
        name="invoice-classification",
        input={
            "invoice_description": invoice_description,
            "vendor_name": vendor_name,
        },
    ) as root_span:

        root_span.update_trace(
            input={"invoice": invoice_description, "vendor": vendor_name},
            metadata={"max_turns": MAX_TURNS},
            tags=["coa-agent"],
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"Please classify this invoice:\n\n"
                            f"{invoice_description}"
                            + (f"\n\nVendor: {vendor_name}" if vendor_name else "")
                        )
                    }
                ],
            }
        ]

        system_prompt = _build_system_prompt(invoice_description, vendor_name)
        turn = 0

        while turn < MAX_TURNS:
            turn += 1

            # ── LLM call ──────────────────────────────────────────────────────
            with langfuse.start_as_current_observation(
                as_type="generation",
                name=f"bedrock-turn-{turn}",
                model=MODEL_ID,
                input=messages,
            ) as gen_span:
                try:
                    response = bedrock_client.converse(
                        modelId=MODEL_ID,
                        system=[{"text": system_prompt}],
                        messages=messages,
                        toolConfig=TOOL_CONFIG,
                        inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
                    )
                except ClientError as e:
                    raise RuntimeError(f"Bedrock API error on turn {turn}: {e}") from e

                output_message = response["output"]["message"]
                stop_reason    = response["stopReason"]

                # Log token usage if available
                usage = response.get("usage", {})
                if usage:
                    gen_span.update(
                        usage={
                            "input":  usage.get("inputTokens", 0),
                            "output": usage.get("outputTokens", 0),
                        },
                        output=output_message,
                    )

            # Add assistant turn to history
            messages.append(output_message)

            # ── Tool use ──────────────────────────────────────────────────────
            if stop_reason == "tool_use":
                tool_results = []

                for block in output_message.get("content", []):
                    # Bedrock wraps tool calls in {"toolUse": {...}} blocks
                    tool_block = block.get("toolUse")
                    if tool_block is None:
                        continue

                    tool_use_id = tool_block["toolUseId"]
                    tool_name   = tool_block["name"]
                    tool_input  = tool_block["input"]

                    # ── Tool span ─────────────────────────────────────────────
                    with langfuse.start_as_current_observation(
                        as_type="span",
                        name=f"tool:{tool_name}",
                        input={"tool": tool_name, "input": tool_input},
                    ) as tool_span:
                        result = execute_tool(tool_name, tool_input, memory, EXPENSES_TREE)
                        tool_span.update(output=result)

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content":   [{"text": json.dumps(result)}],
                            "status":    "error" if "error" in result else "success",
                        }
                    })

                    # Check for early termination after select_leaf
                    if tool_name == "select_leaf" and is_done(memory):
                        root_span.update_trace(
                            output={"selected_leaf": memory.selected_leaf},
                            metadata={
                                "turns": turn,
                                "full_path": memory.selected_path,
                                "log_entries": len(memory.log),
                            },
                        )
                        return memory.selected_leaf  # type: ignore[return-value]

                # Feed tool results back to the model
                messages.append({"role": "user", "content": tool_results})

            # ── End turn (model done talking) ─────────────────────────────────
            else:
                # Model finished without calling select_leaf — extract any text
                final_text = " ".join(
                    block.get("text", "")
                    for block in output_message.get("content", [])
                    if "text" in block
                ).strip()

                # If memory has a selection recorded (e.g. tool call happened last turn)
                if is_done(memory):
                    root_span.update_trace(output={"selected_leaf": memory.selected_leaf})
                    return memory.selected_leaf  # type: ignore[return-value]

                # Model ended without a selection — this is unexpected
                root_span.update_trace(
                    output={"error": "Model ended without calling select_leaf", "text": final_text},
                    level="WARNING",
                )
                raise RuntimeError(
                    f"Agent ended without selecting a leaf after {turn} turns. "
                    f"Last response: {final_text[:300]}"
                )

        # MAX_TURNS exceeded
        root_span.update_trace(
            output={"error": f"MAX_TURNS ({MAX_TURNS}) exceeded without selection"},
            level="ERROR",
        )
        raise RuntimeError(
            f"Agent exceeded MAX_TURNS ({MAX_TURNS}) without calling select_leaf. "
            f"Exploration log has {len(memory.log)} entries."
        )


# ── Batch runner ──────────────────────────────────────────────────────────────

def classify_batch(
    invoices: List[Dict[str, Any]],
    max_workers: int = 10,
) -> Dict[str, Any]:
    """
    Classify a list of invoices in parallel.

    Each invoice dict should have:
      - id: str              — unique identifier
      - description: str     — invoice text
      - vendor: str (optional)

    Returns:
      {
        invoice_id: {
          "selected_leaf": str,
          "error": str | None
        }
      }
    """
    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(
                run_agent,
                inv["description"],
                inv.get("vendor"),
            ): inv["id"]
            for inv in invoices
        }

        for future in as_completed(future_to_id):
            inv_id = future_to_id[future]
            try:
                leaf = future.result()
                results[inv_id] = {"selected_leaf": leaf, "error": None}
            except Exception as e:
                results[inv_id] = {"selected_leaf": None, "error": str(e)}

    return results


# ── CLI for quick testing ─────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_INVOICES = [
        "16/05/2025 Local travel at site - Hotel to Site To and fro - AMC/SAS Site Visit",
        "Monthly internet bill - office connection renewal",
        "Salary advance for Dinesh Marne - May 2025",
    ]

    print("COA Classification Agent — Test Run")
    print("=" * 50)
    for desc in TEST_INVOICES:
        print(f"\nInvoice : {desc}")
        try:
            result = run_agent(desc)
            print(f"→ Ledger : {result}")
        except Exception as e:
            print(f"→ ERROR  : {e}")
    print("\nDone.")

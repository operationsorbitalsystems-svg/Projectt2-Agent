---

## Complete Architecture Document

### The Tools — Precise and Surgical (6 Tools)

Here's the refined set, incorporating your "navigate by exact path" idea:

---

**Tool 1: `get_children(path: list[str])`**

Input: exact path tuple, e.g. `["Expenses", "Indirect Expenses"]`

Returns:
```json
{
  "node": "Indirect Expenses",
  "children": [
    {"name": "Establishment Expenses", "type": "folder", "state": "UNEXPLORED", "leaf_count": 26},
    {"name": "Manufacturing Expenses", "type": "folder", "state": "IN_PROGRESS", "leaf_count": 59},
    {"name": "Other Indirect Expenses", "type": "folder", "state": "EXHAUSTED", "leaf_count": 17}
  ]
}
```

This is always the first call at any node. Agent sees options + their states + leaf counts so it doesn't waste turns entering nodes it already exhausted.

---

**Tool 2: `navigate_to(path: list[str])`**

Input: any path that already exists in `node_states` in memory (tool enforces this — if path not in memory, returns error: "You have not seen this path yet")

This replaces `backtrack()`. Instead of stepping up one level at a time, agent can jump directly to `["Expenses"]` or `["Expenses", "Indirect Expenses"]` in one call. Clean, precise, no wasted turns.

Returns:
```json
{
  "navigated_to": ["Expenses", "Indirect Expenses"],
  "current_node": "Indirect Expenses",
  "unexplored_siblings_here": ["Other Indirect Expenses"]
}
```

---

**Tool 3: `update_node_state(path: list[str], state: str)`**

States allowed: `IN_PROGRESS`, `EXHAUSTED`, `DISCARDED`

Agent explicitly marks nodes. Separation of concerns — navigation and state management are separate actions, not bundled. This also means when agent discards 5 siblings at once, it calls this 5 times (or once with a list — see below).

Actually, let's make it batch: **`update_node_states(updates: list[{path, state}])`** — agent can discard multiple siblings in one tool call.

Returns:
```json
{"updated": [["Expenses", "Depreciation"], ["Expenses", "Tax Expenses"]]}
```

---

**Tool 4: `get_leaf_nodes(path: list[str])`**

Input: any folder path

Returns all leaf node names under it (flat list). The agent uses this when it's confident it's in the right subtree and wants to pick the exact leaf without navigating level by level. For small subtrees (< 10 leaves) the agent can just pick directly.

Returns:
```json
{
  "path": ["Expenses", "Indirect Expenses", "Travelling Expenses"],
  "leaf_count": 3,
  "leaves": ["Local Conveyance", "Staff Conveyance (Office to Factory)", "Travelling Expenses at Site"]
}
```

This is powerful — once the agent has narrowed to a small folder, it doesn't need to keep drilling. It calls this and picks directly.

---

**Tool 5: `get_unexplored_paths()`**

Returns a summary of the current exploration state — specifically, all paths that are `UNEXPLORED` or have unexplored children. No arguments needed, reads from memory.

```json
{
  "unexplored_subtrees": [
    {"path": ["Expenses", "Tax Expenses"], "leaf_count": 3},
    {"path": ["Expenses", "Indirect Expenses", "Other Indirect Expenses"], "leaf_count": 17}
  ],
  "in_progress": [
    {"path": ["Expenses", "Indirect Expenses"], "leaf_count": 102}
  ]
}
```

This is the anti-loop tool. When the agent isn't sure what's left to try, it calls this instead of re-exploring or guessing. Completely eliminates the "spinning in circles" failure mode.

---

**Tool 6: `select_leaf(path: list[str], leaf_name: str)`**

Terminal action. Tool verifies:
1. The path exists in memory
2. `leaf_name` is actually a child of that path
3. That child is actually a leaf node (`value == []`)

If verification fails, returns an error explaining why. Forces agent to be honest about its final answer.

Returns:
```json
{"selected": "Travelling Expenses at Site", "full_path": ["Expenses", "Indirect Expenses", "Travelling Expenses", "Travelling Expenses at Site"], "status": "DONE"}
```

Once this is called, the agent loop terminates.

---

### Memory Structure (Final)

```python
@dataclass
class AgentMemory:
    # Immutable context
    invoice_description: str
    vendor_name: Optional[str]
    
    # The expenses subtree — shared reference, NOT copied per invoice
    # Passed in at construction, never modified
    expenses_tree: Dict[str, Any]  # reference to global tree
    
    # Navigation
    current_path: List[str]  # e.g. ["Expenses", "Indirect Expenses"]
    
    # State of every node the agent has SEEN
    # Key: tuple of path, Value: state string
    node_states: Dict[tuple, str]  # "UNEXPLORED" | "IN_PROGRESS" | "EXHAUSTED" | "DISCARDED"
    
    # Final answer
    selected_leaf: Optional[str] = None
    selected_path: Optional[List[str]] = None
    
    # Audit log
    log: List[Dict]  # list of {action, path, reason, timestamp}
```

**Initial state on construction:**
```python
memory.node_states = {
    ("Expenses",): "IN_PROGRESS",
    ("Expenses", "Depreciation"): "UNEXPLORED",
    ("Expenses", "Direct Expenses"): "UNEXPLORED",
    ("Expenses", "Employee Benefit Expenses"): "UNEXPLORED",
    ("Expenses", "Finance Cost"): "UNEXPLORED",
    ("Expenses", "Indirect Expenses"): "UNEXPLORED",
    ("Expenses", "Purchase Accounts"): "UNEXPLORED",  # leaf
    ("Expenses", "Tax Expenses"): "UNEXPLORED",
}
memory.current_path = ["Expenses"]
```

The tree is pre-seeded with only the top level visible. Deeper nodes get added to `node_states` only when `get_children` is called on their parent — the agent only knows what it has seen.

---

### System Prompt (Skeleton)

```
You are a COA navigation agent. Your job is to classify an invoice by finding 
the single most appropriate leaf node in the Expenses tree.

Invoice: {invoice_description}
Vendor: {vendor_name}

RULES:
1. Always call get_children before entering any new node.
2. Use get_leaf_nodes when you believe you are in the right subtree — 
   if there are fewer than 12 leaves, just pick directly.
3. When you realize you are in the wrong branch, call navigate_to with 
   the correct path from memory — do NOT try paths you haven't seen yet.
4. If you are unsure what remains to explore, call get_unexplored_paths.
5. Mark nodes DISCARDED immediately when you decide by name alone they 
   are irrelevant. Mark them EXHAUSTED when you enter and find nothing.
6. You may ONLY call select_leaf when you are certain. It is your final action.
7. Never enter a node marked EXHAUSTED or DISCARDED.
```

---

### Langfuse v3 Integration

The v3 SDK is OTEL-based. The pattern for our use case:

```python
from langfuse import get_client

langfuse = get_client()  # initialized once at app startup

def classify_invoice(invoice_desc, vendor_name, expenses_tree):
    # One Langfuse trace per invoice
    with langfuse.start_as_current_observation(
        as_type="span",
        name="invoice-classification",
        input={"invoice": invoice_desc, "vendor": vendor_name}
    ) as trace:
        trace.update(metadata={"session_type": "coa_navigation"})
        
        memory = AgentMemory(invoice_desc, vendor_name, expenses_tree)
        result = run_agent(memory)  # the Bedrock agentic loop
        
        trace.update(output={"selected_leaf": result})
    
    return result
```

Inside `run_agent`, each tool call and each LLM call gets its own nested span:

```python
# Inside run_agent, when a tool is executed:
with langfuse.start_as_current_observation(
    as_type="span",
    name=f"tool:{tool_name}",
    input=tool_input
) as tool_span:
    result = execute_tool(tool_name, tool_input, memory)
    tool_span.update(output=result)

# Each Bedrock call:
with langfuse.start_as_current_observation(
    as_type="generation",
    name="bedrock-converse",
    model="amazon.nova-pro-v1:0",  # or whichever
    input=messages
) as gen_span:
    response = bedrock_client.converse(...)
    gen_span.update(output=response)
```

In Langfuse UI you'll see per-invoice: full tree navigation trace, every tool call with args and results, every LLM turn, the exploration log, total tokens used, and the final selected leaf. For 100 invoices running in parallel, each shows up as its own trace with a `session_id` grouping the whole batch.

---

### Parallelism

```python
# At app startup — shared, read-only
expenses_tree = load_expenses_tree(coa_json_path)

# Per batch request
async def classify_batch(invoices: List[Invoice]) -> List[str]:
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(classify_invoice, inv.description, inv.vendor, expenses_tree): inv
            for inv in invoices
        }
        results = {}
        for future, inv in futures.items():
            results[inv.id] = future.result()
    return results
```

`max_workers=15` is safe for Bedrock's rate limits. Each invoice is fully isolated — separate memory, separate message history, separate Langfuse trace.

```
coa_agent/
├── .env
├── config.py
├── memory.py
├── tree_utils.py
├── tools.py
└── agent.py
```

---

### `.env`
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=ap-south-1
BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=
COA_JSON_PATH=./COA_parsed.json
```

---

### `config.py`
- Load `.env` via `python-dotenv`
- Initialize and export `bedrock_client` (boto3)
- Initialize and export `langfuse` client
- Export all env vars as constants (`REGION`, `MODEL_ID`, etc.)
- Load `COA_parsed.json` once, extract `Expenses` subtree, export as `EXPENSES_TREE`

---

### `memory.py`
- `AgentMemory` dataclass with fields:
  - `invoice_description: str`
  - `vendor_name: Optional[str]`
  - `current_path: List[str]`
  - `node_states: Dict[tuple, str]`
  - `selected_leaf: Optional[str]`
  - `selected_path: Optional[List[str]]`
  - `log: List[Dict]`
- `initialize_memory(invoice_description, vendor_name) -> AgentMemory` — seeds top-level Expenses children as UNEXPLORED
- `log_action(memory, action, path, reason)` — appends to memory log
- `is_done(memory) -> bool` — checks if selected_leaf is set

---

### `tree_utils.py`
- `extract_leaf_nodes(subtree) -> List[str]` — recursively get all leaf node names
- `find_matching_non_leaf_node(tree, pattern) -> Optional[Tuple[str, dict]]` — regex search for non-leaf
- `get_node_at_path(tree, path) -> Any` — traverse tree by path list, return node
- `get_children_of(tree, path) -> Dict[str, Any]` — return immediate children dict at a path
- `is_leaf(node) -> bool` — check if node value is `[]`
- `count_leaves(subtree) -> int` — count all leaves under a subtree

---

### `tools.py`
Two sections:

**Implementations** — functions that take `(tool_input: dict, memory: AgentMemory, tree: dict)`:
- `tool_get_children(...)` — returns children with their type, state, leaf count
- `tool_navigate_to(...)` — validates path is in node_states, updates current_path
- `tool_update_node_states(...)` — batch update states in memory
- `tool_get_leaf_nodes(...)` — returns flat leaf list under a path
- `tool_get_unexplored_paths(...)` — scans node_states for UNEXPLORED/IN_PROGRESS
- `tool_select_leaf(...)` — validates and finalizes answer
- `execute_tool(tool_name, tool_input, memory, tree)` — dispatcher

**Bedrock Schema** — `TOOL_CONFIG` dict with all 6 tool specs in Bedrock `toolSpec` format

---

### `agent.py`
- `SYSTEM_PROMPT` — the navigation rules template (filled with invoice + vendor)
- `run_agent(invoice_description, vendor_name) -> str` — main function:
  - Initializes `AgentMemory`
  - Opens Langfuse trace
  - Runs Bedrock `converse` loop
  - On `tool_use` stop reason → dispatches to `execute_tool`, feeds result back
  - On `end_turn` → extracts final text
  - Logs each LLM call and tool call as nested Langfuse spans
  - Returns `selected_leaf`
- `classify_batch(invoices) -> Dict` — ThreadPoolExecutor wrapper for parallel runs


# 🧾 Invoice Ledger Categorization Agent

An AI agent that automatically maps invoice line items to the correct **Expense Ledger** from a company's Chart of Accounts (COA) — powered by **AWS Bedrock** with tool use (function calling).

---

## 🧠 What This Does

Given an invoice with line items like:

```
- "Microsoft Azure monthly subscription - $320"
- "Office cleaning services - $150"
- "Legal retainer fee - $5000"
```

The agent navigates the COA hierarchy using tools, identifies the correct leaf-node expense ledger (e.g., `"Microsoft Azure"`, `"Housekeeping Expenses"`, `"Legal Fees"`), and returns a **constrained, structured response** — only valid ledger names from the COA, never hallucinated ones.

---

## 📁 Project Context

### What is a COA?
A **Chart of Accounts (COA)** is a hierarchical tree of financial ledger categories used in accounting software like **Tally ERP**. It has top-level nodes like:

```
Assets
Liabilities
Income
Expenses   <-- We only care about this
```

The `Expenses` node contains nested sub-groups and, at the deepest level, **leaf nodes** — the actual bookable ledger accounts (e.g., `"Salary Expenses"`, `"AWS Cloud Hosting"`, `"Office Rent"`).

A real COA might have **495 total ledger entries**, but the `Expenses` subtree typically has ~100–150 leaf nodes. The agent only works with those.

### Input Format (COA JSON)
```json
{
  "metadata": { "total_ledgers": 495, ... },
  "hierarchy": {
    "Assets": { ... },
    "Expenses": {
      "Administrative Expenses": {
        "Office Supplies": [],
        "Postage & Courier": []
      },
      "Technology Expenses": {
        "Cloud Hosting": {
          "Microsoft Azure": [],
          "AWS": []
        },
        "Software Subscriptions": []
      }
    }
  }
}
```

Leaf nodes are arrays `[]` — they have no children.

---

## 🏗️ Architecture

### Agent Design: Bedrock Tool Use (Function Calling)

Instead of one-shotting with all 150+ leaf node names crammed into a prompt, the agent uses **two tools** to navigate the COA tree interactively:

```
Tool 1: get_expense_leaf_nodes()
  → Returns ALL leaf node names under the Expenses subtree
  → Used when the agent wants the full allowed list

Tool 2: get_children(node_name: str)
  → Returns immediate children of a given COA node
  → Used to explore the tree top-down before committing
```

The agent reasons about the line item, optionally explores the tree, then picks the correct leaf node.

### Why Tools Instead of One-Shot?
| Approach | Problem |
|---|---|
| One-shot with all 150 ledgers in prompt | Noisy, model may hallucinate or pick wrong one |
| Agent with tools | Model reasons step-by-step, explores relevant subtree first |
| Tools + constrained output schema | Output is **guaranteed** to be a valid ledger name |

### Output Enforcement via Pydantic
The final answer is enforced using a **dynamically generated Pydantic model** with a `Literal` type:

```python
# Dynamically creates: ledger: Literal["Microsoft Azure", "AWS", "Office Rent", ..., "NOT_FOUND"]
OllamaLedger = custom_ledger(leaf_node_list)
```

The JSON schema of this model is appended to the prompt, forcing Bedrock to return only valid ledger names.

---

## 🔁 Agent Flow

```
INPUT: Invoice line item (e.g., "Monthly Azure subscription")
  │
  ▼
SYSTEM PROMPT: Senior CA persona, rules, output format
  │
  ▼
BEDROCK (claude-sonnet / llama / etc.) ← Tool calls if needed
  │   ├── call get_expense_leaf_nodes()  → ["Microsoft Azure", "AWS", ...]
  │   └── call get_children("Technology Expenses") → ["Cloud Hosting", "Software Subscriptions"]
  │
  ▼
MODEL picks best match from leaf nodes
  │
  ▼
OUTPUT: { "ledger": "Microsoft Azure" }  ← Constrained by Pydantic Literal schema
```

---

## 🛠️ Tool Definitions (Bedrock ToolConfig Format)

The tools are passed to Bedrock using the `toolConfig` parameter in the Converse API:

```python
tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_expense_leaf_nodes",
                "description": "Returns all leaf node ledger names under the Expenses section of the Chart of Accounts. Use this to get the complete list of valid expense ledger names.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_children",
                "description": "Returns the immediate children of a given node in the COA hierarchy. Use this to explore the expense tree before selecting a ledger.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "node_name": {
                                "type": "string",
                                "description": "The exact name of the COA node whose children you want."
                            }
                        },
                        "required": ["node_name"]
                    }
                }
            }
        }
    ]
}
```

Pass this as `toolConfig=tool_config` in the Bedrock `converse()` call.

---

## 🔧 Implementation Guide

### Step 1: Tool Handler Functions

```python
import json
from typing import Dict, Any

def get_expense_leaf_nodes(coa_hierarchy: dict) -> list[str]:
    """Extract all leaf nodes from the Expenses subtree."""
    import re
    pattern = re.compile(r'(?i)\bexpense(s)?\b')
    
    expenses = None
    for key, value in coa_hierarchy.items():
        if pattern.fullmatch(key.strip()):
            expenses = value
            break
    
    if not expenses:
        return []
    
    return _extract_leaves(expenses)

def _extract_leaves(node: dict | list, leaves: list = None) -> list[str]:
    if leaves is None:
        leaves = []
    if isinstance(node, list):
        return leaves  # This IS a leaf node (empty list)
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, list) and len(value) == 0:
                leaves.append(key)  # Leaf!
            else:
                _extract_leaves(value, leaves)
    return leaves

def get_children(node_name: str, coa_hierarchy: dict) -> list[str]:
    """Find a node by name and return its immediate children."""
    result = _find_node(node_name, coa_hierarchy)
    if result is None:
        return []
    if isinstance(result, dict):
        return list(result.keys())
    return []

def _find_node(target: str, node: dict | list):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == target:
                return value
            found = _find_node(target, value)
            if found is not None:
                return found
    return None
```

### Step 2: Pydantic Schema Enforcement

```python
from pydantic import BaseModel, Field, create_model
from typing import Literal, List

NOT_FOUND = "NOT_FOUND"

def custom_ledger(leaf_node_list: List[str]):
    """Dynamically create a Pydantic model constrained to valid ledger names."""
    leaf_node_list = leaf_node_list + [NOT_FOUND]
    ledger_literal = Literal[tuple(leaf_node_list)]
    
    OllamaLedger = create_model(
        'OllamaLedger',
        ledger=(ledger_literal, Field(description="Selected ledger account"))
    )
    return OllamaLedger
```

### Step 3: Agent Loop with Bedrock

```python
import boto3
import asyncio
import json
from typing import Optional

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

SYSTEM_PROMPT = """
<role>
You are a Senior Chartered Accountant specializing in Tally ERP categorization for Indian businesses.
Your expertise lies in mapping raw invoice descriptions to specific Chart of Accounts (COA) ledgers.
</role>

<task>
Match the provided invoice line item to the MOST appropriate Expense Ledger.
Use the available tools to explore the Chart of Accounts before making your final selection.
</task>

<rules>
1. STRICT MATCH: Return only an exact ledger name from the COA leaf nodes.
2. SPECIFICITY: Prefer specific ledgers (e.g., "Microsoft Azure") over generic ones (e.g., "Software Expenses").
3. SEMANTIC ALIGNMENT: Match the intent of the expense.
4. NO_MATCH_PROTOCOL: If no ledger fits, return NOT_FOUND.
5. NO INVENTIONS: Never hallucinate or modify ledger names.
</rules>
"""

async def run_ledger_agent(
    line_item: str,
    coa_hierarchy: dict,
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_tool_rounds: int = 5
) -> str:
    """
    Agentic loop: lets Bedrock call tools until it's ready to give a final answer.
    Returns the matched ledger name string.
    """
    # Pre-compute leaf nodes for schema enforcement
    leaf_nodes = get_expense_leaf_nodes(coa_hierarchy)
    LedgerModel = custom_ledger(leaf_nodes)
    json_schema = LedgerModel.model_json_schema()

    # Build tool config
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_expense_leaf_nodes",
                    "description": "Returns all valid expense ledger names (leaf nodes) from the Chart of Accounts.",
                    "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}
                }
            },
            {
                "toolSpec": {
                    "name": "get_children",
                    "description": "Returns immediate children of a COA node. Use to explore the tree.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "node_name": {"type": "string", "description": "Name of the COA node"}
                            },
                            "required": ["node_name"]
                        }
                    }
                }
            }
        ]
    }

    # Initial message
    messages = [
        {
            "role": "user",
            "content": [{
                "text": f"Categorize this invoice line item: '{line_item}'\n\n"
                        f"Return a JSON object matching this schema:\n{json.dumps(json_schema)}"
            }]
        }
    ]

    # Agentic loop
    for round_num in range(max_tool_rounds):
        response = await asyncio.to_thread(
            bedrock_client.converse,
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": 512, "temperature": 0.1}
        )

        output_message = response["output"]["message"]
        messages.append({"role": "assistant", "content": output_message["content"]})

        stop_reason = response.get("stopReason", "")

        # Model wants to use a tool
        if stop_reason == "tool_use":
            tool_results = []
            for block in output_message["content"]:
                if block.get("type") == "toolUse":
                    tool_name = block["name"]
                    tool_input = block.get("input", {})
                    tool_use_id = block["toolUseId"]

                    # Execute the tool
                    if tool_name == "get_expense_leaf_nodes":
                        result = get_expense_leaf_nodes(coa_hierarchy)
                        result_text = json.dumps(result)
                    elif tool_name == "get_children":
                        result = get_children(tool_input.get("node_name", ""), coa_hierarchy)
                        result_text = json.dumps(result)
                    else:
                        result_text = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    tool_results.append({
                        "toolUseId": tool_use_id,
                        "content": [{"text": result_text}]
                    })

            # Feed results back
            messages.append({
                "role": "user",
                "content": [{"toolResult": tr} for tr in tool_results]
            })

        # Model is done — extract final answer
        elif stop_reason == "end_turn":
            for block in output_message["content"]:
                if block.get("type") == "text":
                    text = block["text"].strip()
                    try:
                        # Strip markdown code fences if present
                        clean = text.replace("```json", "").replace("```", "").strip()
                        parsed = json.loads(clean)
                        return parsed.get("ledger", NOT_FOUND)
                    except json.JSONDecodeError:
                        pass
            break

    return NOT_FOUND
```

### Step 4: Batch Processing

```python
async def categorize_invoice(invoice_line_items: list[str], coa_data: dict) -> list[dict]:
    """Categorize all line items in an invoice."""
    hierarchy = coa_data.get("hierarchy", {})
    
    results = []
    for item in invoice_line_items:
        ledger = await run_ledger_agent(item, hierarchy)
        results.append({
            "line_item": item,
            "ledger": ledger,
            "matched": ledger != NOT_FOUND
        })
        print(f"  ✅ '{item}' → '{ledger}'")
    
    return results

# Example usage
async def main():
    with open("COA.json") as f:
        coa_data = json.load(f)
    
    line_items = [
        "Monthly Microsoft Azure subscription - compute instances",
        "Office cleaning and housekeeping services",
        "Legal retainer fee - corporate compliance",
        "Team lunch - client meeting",
    ]
    
    results = await categorize_invoice(line_items, coa_data)
    print(json.dumps(results, indent=2))

asyncio.run(main())
```

---

## 📂 Expected File Structure

```
project/
├── README.md                  ← This file
├── agent.py                   ← Main agent loop (run_ledger_agent)
├── tools.py                   ← get_expense_leaf_nodes, get_children
├── schema.py                  ← custom_ledger Pydantic factory
├── bedrock_client.py          ← Bedrock call_bedrock() wrapper (async)
├── config.py                  ← AWS creds, model ID, semaphore
├── utils/
│   └── logger.py              ← setup_logger()
├── data/
│   └── COA.json               ← Parsed chart of accounts
└── main.py                    ← Entry point / batch runner
```

---

## ⚙️ Configuration

In `config.py`:

```python
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# or: "meta.llama3-70b-instruct-v1:0"
# or: "amazon.nova-pro-v1:0"

AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."

import asyncio
bedrock_semaphore = asyncio.Semaphore(5)  # Max concurrent Bedrock calls
```

---

## 🧪 Quick Test

```python
# Minimal smoke test
import asyncio, json

with open("data/COA.json") as f:
    coa = json.load(f)

result = asyncio.run(run_ledger_agent(
    "AWS EC2 monthly hosting charges",
    coa["hierarchy"]
))
print(result)  # Expected: something like "Cloud Hosting Expenses" or "AWS"
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Agent + tools instead of one-shot | Model can reason about ambiguous items by exploring the tree |
| Only expose `Expenses` leaf nodes | Avoids noise from 495 total ledgers; invoices map to expenses only |
| Pydantic `Literal` schema enforcement | Hard constraint — model physically cannot return invalid ledger names |
| `NOT_FOUND` as valid literal option | Prevents forced wrong matches; signals human review needed |
| Bedrock Converse API | Works across all model families (Claude, Llama, Nova) with same interface |
| Async + semaphore | Safe for concurrent invoice processing without hitting rate limits |

---

## ❗ Common Pitfalls

- **Tool `inputSchema` must be valid JSON Schema** — Bedrock will reject malformed schemas silently.
- **Parse `stopReason`** — don't assume the model is done after one turn; check for `"tool_use"` vs `"end_turn"`.
- **Model may return JSON inside markdown fences** — always strip ` ```json ``` ` before `json.loads()`.
- **`leaf_nodes` list must be computed fresh per COA** — different companies have different ledger trees.
- **`NOT_FOUND` must be appended to the Literal list** — otherwise the model has no escape hatch for bad matches.
"""
tools.py
────────
Two sections:
  1. Tool implementations  — functions called during agent execution
  2. TOOL_CONFIG           — Bedrock converse() toolSpec schema for all 6 tools

Tools:
  get_children        — see immediate children of a node with their states
  navigate_to         — jump to any previously seen path
  update_node_states  — batch-mark nodes as IN_PROGRESS / EXHAUSTED / DISCARDED
  get_leaf_nodes      — list all leaves under a path (for final selection)
  get_unexplored_paths — anti-loop: show what is still left to try
  select_leaf         — terminal action, validated answer
"""

from typing import Any, Dict, List, Optional

from memory import (
    AgentMemory,
    DISCARDED,
    EXHAUSTED,
    IN_PROGRESS,
    SELECTED,
    UNEXPLORED,
    ensure_path_known,
    get_state,
    log_action,
    set_state,
)
from tree_utils import (
    count_leaves,
    describe_children,
    extract_leaf_nodes,
    get_children_of,
    get_node_at_path,
    is_leaf,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def tool_get_children(
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return immediate children of the node at `path`, enriched with:
      - type (leaf / folder)
      - current state from memory
      - leaf_count

    Also seeds any NEW children into node_states as UNEXPLORED.
    """
    path: List[str] = tool_input["path"]

    # Validate path exists in tree
    try:
        children_info = describe_children(tree, path)
    except (KeyError, TypeError) as e:
        return {"error": str(e)}

    # Enrich with state from memory; seed new children as UNEXPLORED
    enriched = []
    for child in children_info:
        child_path = path + [child["name"]]
        child_path_tuple = tuple(child_path)

        # Seed into memory if first time seeing this child
        if child_path_tuple not in memory.node_states:
            memory.node_states[child_path_tuple] = UNEXPLORED

        enriched.append({
            **child,
            "state": memory.node_states[child_path_tuple],
        })

    log_action(memory, "GET_CHILDREN", path, f"Saw {len(enriched)} children.")
    return {
        "node": path[-1] if path else "root",
        "path": path,
        "children": enriched,
    }


def tool_navigate_to(
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Move current_path to the given path.

    Rules:
      - Path must already be in node_states (agent must have seen it before).
      - Cannot navigate to an EXHAUSTED or DISCARDED node.
      - Marks the destination as IN_PROGRESS.
    """
    path: List[str] = tool_input["path"]

    if not ensure_path_known(memory, path):
        return {
            "error": (
                f"Path {path} has not been seen yet. "
                "Call get_children on the parent first."
            )
        }

    state = get_state(memory, path)
    if state in (EXHAUSTED, DISCARDED):
        return {
            "error": (
                f"Cannot navigate to '{path[-1]}' — it is already marked {state}. "
                "Choose an UNEXPLORED path instead."
            )
        }

    # Validate the path actually exists in the tree
    try:
        get_node_at_path(tree, path)
    except KeyError as e:
        return {"error": f"Tree lookup failed: {e}"}

    memory.current_path = path[:]
    set_state(memory, path, IN_PROGRESS)
    log_action(memory, "NAVIGATE_TO", path)

    # Show unexplored siblings at the parent level as a helpful hint
    parent_path = path[:-1]
    unexplored_siblings = [
        p[-1]
        for p, s in memory.node_states.items()
        if list(p[:-1]) == parent_path and s == UNEXPLORED and list(p) != path
    ]

    return {
        "navigated_to": path,
        "current_node": path[-1],
        "unexplored_siblings_at_parent": unexplored_siblings,
    }


def tool_update_node_states(
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Batch update node states.

    Input:
      updates: list of {path: [...], state: "EXHAUSTED"|"DISCARDED"|"IN_PROGRESS"}

    SELECTED is not allowed here — use select_leaf for that.
    """
    updates: List[Dict[str, Any]] = tool_input["updates"]
    updated = []
    errors = []

    allowed = {EXHAUSTED, DISCARDED, IN_PROGRESS}

    for item in updates:
        path: List[str] = item["path"]
        state: str = item["state"]

        if state not in allowed:
            errors.append(
                f"State '{state}' not allowed via update_node_states. "
                f"Allowed: {allowed}. Use select_leaf to mark SELECTED."
            )
            continue

        # Seed path if not yet in memory (allows agent to pre-discard unseen siblings)
        if not ensure_path_known(memory, path):
            memory.node_states[tuple(path)] = UNEXPLORED

        set_state(memory, path, state)
        log_action(memory, f"STATE→{state}", path)
        updated.append(path)

    result: Dict[str, Any] = {"updated": updated}
    if errors:
        result["errors"] = errors
    return result


def tool_get_leaf_nodes(
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a flat list of all leaf node names under the given path.

    Use this when the agent is confident it is in the right subtree
    and wants to pick the exact leaf without further drilling.
    """
    path: List[str] = tool_input["path"]

    try:
        subtree = get_node_at_path(tree, path)
    except KeyError as e:
        return {"error": str(e)}

    if is_leaf(subtree):
        # The node itself is a leaf
        return {
            "path": path,
            "leaf_count": 1,
            "leaves": [path[-1]],
        }

    leaves = extract_leaf_nodes(subtree)
    log_action(memory, "GET_LEAVES", path, f"{len(leaves)} leaves found.")
    return {
        "path": path,
        "leaf_count": len(leaves),
        "leaves": leaves,
    }


def tool_get_unexplored_paths(
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a summary of what remains to be explored.

    Returns:
      - unexplored_subtrees: paths with state UNEXPLORED + their leaf counts
      - in_progress: paths currently being explored
    """
    unexplored = []
    in_progress = []

    for path_tuple, state in memory.node_states.items():
        path = list(path_tuple)
        if state == UNEXPLORED:
            try:
                subtree = get_node_at_path(tree, path)
                lc = count_leaves(subtree) if isinstance(subtree, dict) else 0
            except KeyError:
                lc = -1
            unexplored.append({"path": path, "leaf_count": lc})

        elif state == IN_PROGRESS:
            in_progress.append({"path": path})

    return {
        "unexplored_subtrees": unexplored,
        "in_progress": in_progress,
    }


def tool_select_leaf(
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Terminal action — validate and record the final selected leaf.

    Validates:
      1. parent path exists in memory
      2. leaf_name is an actual child of that path in the tree
      3. that child is actually a leaf node (value == [])
    """
    path: List[str] = tool_input["path"]        # path to the PARENT folder
    leaf_name: str = tool_input["leaf_name"]

    # 1. Parent must be known
    if not ensure_path_known(memory, path):
        return {
            "error": (
                f"Parent path {path} has not been seen. "
                "Call get_children first to confirm this path exists."
            )
        }

    # 2. Get the parent's children from the tree
    try:
        children = get_children_of(tree, path)
    except (KeyError, TypeError) as e:
        return {"error": f"Cannot resolve parent path: {e}"}

    if leaf_name not in children:
        return {
            "error": (
                f"'{leaf_name}' is not a child of {path}. "
                f"Available children: {list(children.keys())}"
            )
        }

    # 3. Must be a leaf
    if not is_leaf(children[leaf_name]):
        return {
            "error": (
                f"'{leaf_name}' is not a leaf node — it still has children. "
                "Navigate deeper and pick a leaf."
            )
        }

    # ✅ All checks pass — record selection
    full_path = path + [leaf_name]
    memory.selected_leaf = leaf_name
    memory.selected_path = full_path
    memory.node_states[tuple(full_path)] = SELECTED

    log_action(memory, "SELECTED", full_path, "Final answer recorded.")

    return {
        "status": "DONE",
        "selected": leaf_name,
        "full_path": full_path,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
    memory: AgentMemory,
    tree: Dict[str, Any],
) -> Dict[str, Any]:
    """Route tool_name to the correct implementation."""
    dispatch = {
        "get_children":         tool_get_children,
        "navigate_to":          tool_navigate_to,
        "update_node_states":   tool_update_node_states,
        "get_leaf_nodes":       tool_get_leaf_nodes,
        "get_unexplored_paths": tool_get_unexplored_paths,
        "select_leaf":          tool_select_leaf,
    }
    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: '{tool_name}'. Available: {list(dispatch.keys())}"}
    return fn(tool_input, memory, tree)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bedrock toolSpec schema
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_children",
                "description": (
                    "Get the immediate children of a node in the Expenses tree. "
                    "Always call this first when entering any node. "
                    "Returns each child's name, type (leaf/folder), current exploration state, "
                    "and how many leaf nodes it contains. "
                    "Use the state field to avoid re-entering EXHAUSTED or DISCARDED nodes."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Exact path from root. "
                                    'Example: ["Expenses"] or ["Expenses", "Indirect Expenses"]'
                                ),
                            }
                        },
                        "required": ["path"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "navigate_to",
                "description": (
                    "Move to any path you have previously seen (it must exist in your memory). "
                    "Use this to enter a child node OR to backtrack to a parent or sibling. "
                    "You cannot navigate to a path marked EXHAUSTED or DISCARDED. "
                    "If you need to jump back to a higher level, pass the shorter path directly."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "The full path to navigate to. Must be a path you have "
                                    "already seen via get_children. "
                                    'Example: ["Expenses", "Indirect Expenses"]'
                                ),
                            }
                        },
                        "required": ["path"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "update_node_states",
                "description": (
                    "Batch-update the exploration state of one or more nodes. "
                    "Use DISCARDED when you decide by name alone that a node is irrelevant — do not enter it. "
                    "Use EXHAUSTED when you have fully explored a node and found nothing suitable. "
                    "Use IN_PROGRESS to mark a node you are currently exploring (usually done automatically by navigate_to). "
                    "You cannot use this to mark a node as SELECTED — use select_leaf for that."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Full path to the node.",
                                        },
                                        "state": {
                                            "type": "string",
                                            "enum": ["IN_PROGRESS", "EXHAUSTED", "DISCARDED"],
                                            "description": "New state for this node.",
                                        },
                                    },
                                    "required": ["path", "state"],
                                },
                                "description": "List of {path, state} updates to apply.",
                            }
                        },
                        "required": ["updates"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_leaf_nodes",
                "description": (
                    "Get a flat list of ALL leaf node names under a given path. "
                    "Use this when you believe you are in the right subtree and want to "
                    "pick the exact ledger without navigating level by level. "
                    "If the subtree has fewer than ~12 leaves, just review the list and call select_leaf directly."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Path to the subtree whose leaves you want. "
                                    'Example: ["Expenses", "Indirect Expenses", "Travelling Expenses"]'
                                ),
                            }
                        },
                        "required": ["path"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_unexplored_paths",
                "description": (
                    "Get a summary of everything that is still left to explore. "
                    "Call this whenever you feel lost, stuck, or unsure what to try next. "
                    "Returns all UNEXPLORED paths (with leaf counts) and IN_PROGRESS paths. "
                    "This prevents going in circles and ensures complete coverage."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "select_leaf",
                "description": (
                    "Your final action. Select the single most appropriate leaf node (ledger account) "
                    "for this invoice. The tool will validate that the leaf exists and is truly a leaf. "
                    "Only call this when you are certain. You cannot undo this action."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Path to the PARENT folder of the leaf — NOT including the leaf name itself. "
                                    'Example: ["Expenses", "Indirect Expenses", "Travelling Expenses"]'
                                ),
                            },
                            "leaf_name": {
                                "type": "string",
                                "description": (
                                    "The exact name of the leaf node to select. "
                                    'Example: "Travelling Expenses at Site"'
                                ),
                            },
                        },
                        "required": ["path", "leaf_name"],
                    }
                },
            }
        },
    ]
}

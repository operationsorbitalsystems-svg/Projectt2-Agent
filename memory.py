"""
memory.py
─────────
AgentMemory dataclass — one instance per invoice classification run.

Node state lifecycle:
  UNEXPLORED  → agent has seen the node name but not entered it yet
  IN_PROGRESS → agent is currently inside this node (path is active)
  EXHAUSTED   → agent entered, found nothing useful, backed out
  DISCARDED   → agent rejected by name alone, never entered
  SELECTED    → terminal state, only for the final chosen leaf node
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Valid states
UNEXPLORED  = "UNEXPLORED"
IN_PROGRESS = "IN_PROGRESS"
EXHAUSTED   = "EXHAUSTED"
DISCARDED   = "DISCARDED"
SELECTED    = "SELECTED"

VALID_STATES = {UNEXPLORED, IN_PROGRESS, EXHAUSTED, DISCARDED, SELECTED}


@dataclass
class AgentMemory:
    # ── Immutable invoice context ─────────────────────────────────────────────
    invoice_description: str
    vendor_name: Optional[str]

    # ── Navigation ────────────────────────────────────────────────────────────
    # Full path from root, e.g. ["Expenses", "Indirect Expenses"]
    current_path: List[str] = field(default_factory=lambda: ["Expenses"])

    # Key: tuple path, Value: one of VALID_STATES
    # Seeded with top-level Expenses children at init time
    node_states: Dict[Tuple[str, ...], str] = field(default_factory=dict)

    # ── Terminal answer ───────────────────────────────────────────────────────
    selected_leaf: Optional[str] = None
    selected_path: Optional[List[str]] = None

    # ── Audit log ─────────────────────────────────────────────────────────────
    log: List[Dict[str, Any]] = field(default_factory=list)


# ── Factory ───────────────────────────────────────────────────────────────────

def initialize_memory(
    expenses_tree: Dict[str, Any],
    invoice_description: str,
    vendor_name: Optional[str] = None,
) -> AgentMemory:
    """
    Create a fresh AgentMemory for one invoice classification run.

    Pre-seeds node_states with:
      - ("Expenses",)          → IN_PROGRESS  (we start here)
      - ("Expenses", <child>)  → UNEXPLORED   for each top-level child
    """
    memory = AgentMemory(
        invoice_description=invoice_description,
        vendor_name=vendor_name,
        current_path=["Expenses"],
        node_states={},
        log=[],
    )

    # Root is already IN_PROGRESS
    memory.node_states[("Expenses",)] = IN_PROGRESS

    # Seed immediate children of Expenses as UNEXPLORED
    for child_key in expenses_tree.keys():
        path_tuple = ("Expenses", child_key)
        memory.node_states[path_tuple] = UNEXPLORED

    log_action(memory, "INIT", ["Expenses"], "Memory initialized, top-level children seeded.")
    return memory


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_action(
    memory: AgentMemory,
    action: str,
    path: List[str],
    reason: str = "",
) -> None:
    """Append a structured entry to the exploration log."""
    memory.log.append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "action": action,
        "path": path[:],        # copy so mutations don't affect the log
        "reason": reason,
    })


def is_done(memory: AgentMemory) -> bool:
    """Returns True once a leaf has been selected."""
    return memory.selected_leaf is not None


def get_state(memory: AgentMemory, path: List[str]) -> Optional[str]:
    """Return the current state of a path, or None if never seen."""
    return memory.node_states.get(tuple(path))


def set_state(memory: AgentMemory, path: List[str], state: str) -> None:
    """Update the state of a node. Raises ValueError for unknown states."""
    if state not in VALID_STATES:
        raise ValueError(f"Unknown state '{state}'. Must be one of {VALID_STATES}.")
    memory.node_states[tuple(path)] = state


def ensure_path_known(memory: AgentMemory, path: List[str]) -> bool:
    """Returns True if the path has been seen (exists in node_states)."""
    return tuple(path) in memory.node_states

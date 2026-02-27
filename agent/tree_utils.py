"""
tree_utils.py
─────────────
Pure functions for navigating the COA tree structure.
No side effects — all functions are stateless and read-only.

Tree shape conventions:
  - dict node  → non-leaf (has children)
  - [] (empty list) → leaf node (final ledger account)
  - non-empty list  → should not appear in well-formed COA, treated as leaf
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# ── Basic node classification ─────────────────────────────────────────────────

def is_leaf(value: Any) -> bool:
    """A node is a leaf if its value is an empty list."""
    return isinstance(value, list) and len(value) == 0


def is_folder(value: Any) -> bool:
    """A node is a folder if its value is a non-empty dict."""
    return isinstance(value, dict) and len(value) > 0


# ── Navigation ────────────────────────────────────────────────────────────────

def get_node_at_path(tree: Dict[str, Any], path: List[str]) -> Any:
    """
    Traverse the tree following path and return the node value.

    IMPORTANT: `tree` is the Expenses sub-dict (already inside "Expenses").
    `path` always starts with "Expenses" as the logical root label, but since
    the tree reference already points inside Expenses, we strip path[0] if it
    equals "Expenses" before traversing.

    Examples:
      path=["Expenses"]                                → returns tree (the root dict)
      path=["Expenses", "Indirect Expenses"]           → returns tree["Indirect Expenses"]
      path=["Expenses", "Indirect Expenses", "Foo"]    → returns tree["Indirect Expenses"]["Foo"]

    Raises KeyError if any segment of the path does not exist.
    """
    if not path:
        return tree

    # Strip the logical root label — tree IS already the Expenses dict
    segments = path[1:] if path[0] == "Expenses" else path

    current = tree
    for segment in segments:
        if not isinstance(current, dict):
            raise KeyError(
                f"Cannot descend into '{segment}': parent is not a dict."
            )
        if segment not in current:
            raise KeyError(
                f"Key '{segment}' not found. Available: {list(current.keys())}"
            )
        current = current[segment]

    return current


def get_children_of(tree: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
    """
    Return the immediate children of the node at path as a dict.

    Each key is the child name.
    Each value is either:
      - {} / dict  → folder
      - []         → leaf

    Raises KeyError if path doesn't exist.
    Raises TypeError if the node at path is not a dict (i.e. it's a leaf).
    """
    node = get_node_at_path(tree, path)
    if not isinstance(node, dict):
        raise TypeError(
            f"Node at {path} is a leaf (value={node!r}), not a folder. Cannot get children."
        )
    return node


# ── Leaf extraction ───────────────────────────────────────────────────────────

def extract_leaf_nodes(subtree: Any) -> List[str]:
    """
    Recursively collect all leaf node names under a given subtree.

    A leaf is any key whose value is [] (empty list).
    Returns a flat list of leaf names.
    """
    leaves: List[str] = []

    def _traverse(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if is_leaf(value):
                    leaves.append(key)
                elif isinstance(value, dict):
                    _traverse(value)
                elif isinstance(value, list):
                    # Non-empty list — traverse items (shouldn't happen in clean COA)
                    for item in value:
                        _traverse(item)
        elif isinstance(node, list):
            for item in node:
                _traverse(item)

    _traverse(subtree)
    return leaves


def count_leaves(subtree: Any) -> int:
    """Count total leaf nodes under a subtree."""
    return len(extract_leaf_nodes(subtree))


# ── Search helpers ────────────────────────────────────────────────────────────

def find_matching_non_leaf_node(
    tree: Dict[str, Any],
    key_pattern: re.Pattern,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Recursively search tree for the first key that:
      - matches key_pattern (regex search)
      - has a non-empty dict as its value (i.e. it's a folder)

    Returns (matched_key, subtree_dict) or None if not found.
    """
    for key, value in tree.items():
        if isinstance(key, str) and key_pattern.search(key):
            if is_folder(value):
                return key, value

        if isinstance(value, dict):
            result = find_matching_non_leaf_node(value, key_pattern)
            if result:
                return result

    return None


def describe_children(
    tree: Dict[str, Any],
    path: List[str],
) -> List[Dict[str, Any]]:
    """
    Return a description of every immediate child at path, including:
      - name: str
      - type: "leaf" | "folder"
      - leaf_count: int  (0 for leaves, n for folders)

    Used by tool_get_children to build its response payload.
    """
    children_dict = get_children_of(tree, path)
    result = []
    for name, value in children_dict.items():
        if is_leaf(value):
            result.append({"name": name, "type": "leaf", "leaf_count": 0})
        else:
            result.append({
                "name": name,
                "type": "folder",
                "leaf_count": count_leaves(value),
            })
    return result


from typing import Dict, Any
from pathlib import Path
from coa_utils.main import parse_coa



# ── Load & extract Expenses subtree ───────────────────────────────────────────
def _load_expenses_tree(path: str) -> Dict[str, Any]:
    """
    Load COA_parsed.json and return the 'Expenses' sub-dict.
    Raises FileNotFoundError if path is wrong.
    Raises KeyError if 'Expenses' is not found in hierarchy.
    """
    

    
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"COA JSON not found at: {resolved}")

    coa_tree = parse_coa(path)
    raw = coa_tree.model_dump(mode="json")
    hierarchy: Dict[str, Any] = raw.get("hierarchy", {})

    # # Case-insensitive match for "Expenses" top-level key
    # pattern = re.compile(r"(?i)^\s*expenses\s*$")
    # for key, value in hierarchy.items():
    #     if pattern.match(key):
    #         if not isinstance(value, dict) or not value:
    #             raise ValueError(f"Expenses key '{key}' is empty or not a dict.")
    #         return value

    return hierarchy

EXPENSES_TREE: Dict[str, Any] = _load_expenses_tree("/home/soham/Documents/orbtl/Hypro/COA.pdf")


import json
from agent.agent import BASE_PATH, JSON_NAME, run_agent
from pathlib import Path

# ── CLI for quick testing ─────────────────────────────────────────────────────
if __name__ == "__main__":
    
    json_path = BASE_PATH + "/" + JSON_NAME
    
    resolved = Path(json_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"COA JSON not found at: {resolved}")

    with open(resolved) as f:
        raw = json.load(f)
        
    line_items_array = raw["line_items"]
    
    total_ledger_narration = ""
    
    for line in line_items_array:
        total_ledger_narration += line["description"] + "\n"
        
    
    
    TEST_INVOICES = [
        # "16/05/2025 Local travel at site - Hotel to Site To and fro - AMC/SAS Site Visit",
        total_ledger_narration
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
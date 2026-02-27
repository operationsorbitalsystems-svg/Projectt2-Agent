import json
from agent.agent import BASE_PATH, JSON_NAME, run_agent
from pathlib import Path
from utils.logger import configure_logging
from config import DEBUG
from mistral_comp.invoice_parser import InvoiceParser
import asyncio

pdf_path  = "/home/soham/Documents/orbtl/Hypro-2/input/45.pdf"

invoice_parser = InvoiceParser()


configure_logging(debug=DEBUG)


async def main():

    # json_path = BASE_PATH + "/" + JSON_NAME
    
    # resolved = Path(json_path).resolve()
    # if not resolved.exists():
    #     raise FileNotFoundError(f"COA JSON not found at: {resolved}")

    # with open(resolved) as f:
    #     raw = json.load(f)
        
    # line_items_array = raw["line_items"]
    
    # total_ledger_narration = ""
    
    # for line in line_items_array:
    #     total_ledger_narration += line["description"] + "\n"
        
    success_or_fail, invoice_data, error_if_fail = await invoice_parser.parse_invoice(
        pdf_path=pdf_path
    )
    
    if success_or_fail:
        
        invoice = invoice_data.model_dump()
        
        vendor_name = invoice_data.header.vendor_name
        
        line_items_array = invoice["line_items"]
        
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
                result = run_agent(desc, vendor_name)
                print(f"→ Ledger : {result}")
            except Exception as e:
                print(f"→ ERROR  : {e}")
        print("\nDone.")
        
    else:
        print("error while mistrakl parsing")



# ── CLI for quick testing ─────────────────────────────────────────────────────
if __name__ == "__main__":
    
    asyncio.run(main())
    
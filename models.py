from pydantic import field_serializer, field_validator
import re
from utils.logger import setup_logger

logger = setup_logger("models")

# === Invoice Data Models ===


class InvoiceHeader(BaseModel):
    invoice_number: str
    invoice_date: str
    vendor_name: str
    vendor_address: Optional[str] = None
    vendor_gstin: Optional[str] = None
    place_of_supply: Optional[str] = None
    vendor_pin_code: Optional[str] = None

    @field_validator("vendor_pin_code")
    @classmethod
    def validate_pincode(cls, v):
        if v is None:
            return None
        
        cleaned = re.sub(r"[ -]", "", v.strip())

        if cleaned.isdigit() and len(cleaned) == 6:
            return cleaned
        
        # Non-blocking behavior
        logger.warning(
            f"Invalid vendor_pin_code detected ('{v}'). "
            "Setting vendor_pin_code to None."
        )
        return None


class InvoiceLineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: float


from pydantic import BaseModel, model_validator
from typing import List, Optional

class InvoiceData(BaseModel):
    header: InvoiceHeader
    line_items: List[InvoiceLineItem]
    subtotal: Optional[float] = None
    cgst_tax_amount: Optional[float] = None
    sgst_tax_amount: Optional[float] = None
    igst_tax_amount: Optional[float] = None
    total_amount: float
    currency: str
    already_recieved: Optional[float] = None

    @model_validator(mode="after")
    def sync_cgst_sgst(self):
        # Case 1: CGST exists but SGST missing
        if self.cgst_tax_amount is not None and self.sgst_tax_amount is None:
            self.sgst_tax_amount = self.cgst_tax_amount

        # Case 2: SGST exists but CGST missing
        elif self.sgst_tax_amount is not None and self.cgst_tax_amount is None:
            self.cgst_tax_amount = self.sgst_tax_amount

        return self
    
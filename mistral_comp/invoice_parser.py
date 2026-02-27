import json
import logging
import base64
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from config import MISTRAL_API_KEY, mistral_semaphore
from models import InvoiceData, InvoiceHeader, InvoiceLineItem
from utils.logger import setup_logger
from mistralai import Mistral
                
logger = setup_logger()


#global_semaphore_initialization



class InvoiceParser:
    """Handles invoice parsing using Mistral OCR with real API integration"""
    
    def __init__(self):
        self.api_key = MISTRAL_API_KEY
        self.client = None
        
        global mistral_semaphore
        
        self.semaphore = mistral_semaphore 
        
        if self.api_key:
            try:

                self.client = Mistral(api_key=self.api_key)
                logger.info("Mistral client initialized successfully")
            except ImportError:
                logger.warning(
                    "mistralai package not found. Install with: "
                    "pip install mistralai --break-system-packages"
                )
                self.client = None
        else:
            logger.warning("MISTRAL_API_KEY not set. Parsing will use mock data.")
    
    @staticmethod
    def encode_pdf_to_base64(pdf_path: str) -> str:
        """Encode PDF file to base64 string"""
        try:
            with open(pdf_path, "rb") as pdf_file:
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding PDF: {str(e)}")
            raise
    
    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Check if error is retryable (500 errors, service unavailable, etc.)"""
        error_str = str(error).lower()
        retryable_indicators = [
            "500", "503", "502", "504",
            "service unavailable",
            "internal server error",
            "bad gateway",
            "gateway timeout",
            "timeout"
        ]
        return any(indicator in error_str for indicator in retryable_indicators)
    
    
    async def parse_invoice(self, pdf_path: str, max_retries: int = 4) -> Tuple[bool, Optional[InvoiceData], Optional[str]]:
        """
        Parse invoice using actual Mistral OCR API with retry logic
        
        Args:
            pdf_path: Path to the PDF file
            max_retries: Maximum number of retries (default: 4)
        
        Returns:
            Tuple of (success, invoice_data, error_message)
        """
        filename = Path(pdf_path).name
        

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retry {attempt}/{max_retries}: {filename}")
                else:
                    logger.info(f"Processing with Mistral OCR: {filename}")
                
                # Encode PDF to base64
                base64_pdf = await asyncio.to_thread(
                    self.encode_pdf_to_base64,
                    pdf_path
                )
                
                # Create JSON schema for structured output
                invoice_schema = InvoiceData.model_json_schema()
                
                async with self.semaphore:
                    # Call Mistral OCR API with structured output
                    ocr_response = await self.client.ocr.process_async(
                        model="mistral-ocr-latest",
                        document={
                            "type": "document_url",
                            "document_url": f"data:application/pdf;base64,{base64_pdf}"
                        },
                        document_annotation_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "InvoiceData",
                                "description": "Structured invoice data extraction",
                                "schema": invoice_schema,
                                "strict": True
                            }
                        }
                    )
                
                
                # Parse document_annotation from response
                if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
                    try:
                        annotation_dict = json.loads(ocr_response.document_annotation)
                        invoice_data = InvoiceData.model_validate(annotation_dict)
                        
                        logger.info(
                            f"✅ Successfully parsed {filename} → "
                            f"Invoice: {invoice_data.header.invoice_number} | "
                            f"Total: {invoice_data.currency} {invoice_data.total_amount:,.2f}"
                        )
                        return True, invoice_data, None
                    
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse Mistral response JSON: {str(e)}"
                        logger.error(error_msg)
                        return False, None, error_msg
                    
                    except Exception as e:
                        error_msg = f"Failed to validate invoice data: {str(e)}"
                        logger.error(error_msg)
                        return False, None, error_msg
                else:
                    raise ValueError("No document_annotation returned from Mistral OCR")
            
            except Exception as e:
                # Check if we should retry
                if attempt < max_retries and self.is_retryable_error(e):
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s
                    logger.warning(
                        f"Retryable error for {filename}, waiting {wait_time}s before "
                        f"retry {attempt + 1}/{max_retries}: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure or non-retryable error
                    error_msg = str(e)
                    if attempt == max_retries:
                        logger.error(f"❌ Failed to parse {filename} after {max_retries} retries: {error_msg}")
                    else:
                        logger.error(f"❌ Non-retryable error for {filename}: {error_msg}")
                    return False, None, error_msg
    
        return False, None, f"Failed to parse {filename} after all retries"
    
    
    def extract_metadata(self, invoice_data: InvoiceData) -> Dict[str, Any]:
        """
        Extract key metadata from parsed invoice
        Returns summary information for quick display
        """
        return {
            "invoice_number": invoice_data.header.invoice_number,
            "vendor_name": invoice_data.header.vendor_name,
            "total_amount": invoice_data.total_amount,
            "currency": invoice_data.currency,
            "line_items_count": len(invoice_data.line_items),
            "invoice_date": invoice_data.header.invoice_date,
        }
    
    def validate_invoice_data(self, invoice_data: InvoiceData) -> Tuple[bool, str]:
        """
        Validate extracted invoice data
        Returns: (is_valid, error_message)
        """
        # Check required fields
        if not invoice_data.header.invoice_number:
            return False, "Missing invoice number"
        
        if not invoice_data.header.vendor_name:
            return False, "Missing vendor name"
        
        if invoice_data.total_amount <= 0:
            return False, "Invalid total amount"
        
        if not invoice_data.currency:
            return False, "Missing currency"
        
        if not invoice_data.line_items:
            return False, "No line items found"
        
        # Validate line items
        for item in invoice_data.line_items:
            if item.amount <= 0:
                return False, f"Invalid amount for item: {item.description}"
        
        return True, ""
    
    def format_invoice_response(self,
                               filename: str,
                               pdf_path: str,
                               json_path: str,
                               invoice_data: Optional[InvoiceData],
                               error: Optional[str] = None,
                               xl_output: Optional[List[dict]] = None) -> Dict[str, Any]:
        """
        Format invoice data into response structure

        Args:
            filename: Invoice filename
            pdf_path: Path to PDF file
            json_path: Path to JSON result file
            invoice_data: Parsed invoice data
            error: Error message if any
            xl_output: XL output rows for journal entry import

        Returns:
            Formatted response dictionary
        """
        from datetime import datetime

        if error or not invoice_data:
            return {
                "filename": filename,
                "pdf_path": pdf_path,
                "json_path": json_path,
                "status": "error",
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            }

        metadata = self.extract_metadata(invoice_data)

        return {
            "filename": filename,
            "pdf_path": pdf_path,
            "json_path": json_path,
            "status": "success",
            "invoice_number": metadata["invoice_number"],
            "vendor_name": metadata["vendor_name"],
            "total_amount": metadata["total_amount"],
            "currency": metadata["currency"],
            "line_items_count": metadata["line_items_count"],
            "xl_output": xl_output,  # XL output for journal entry
            "data": invoice_data.model_dump(),
            "error": None,
            "timestamp": datetime.utcnow().isoformat()
        }
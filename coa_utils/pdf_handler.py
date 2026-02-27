from typing import List, Dict, Tuple, Optional
from .models import Word
import pdfplumber
import fitz
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PDFHandler:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.logger = logger
    
    def extract_all_words(self) -> Dict[int, List[Word]]:
        """Extract all words from all pages with coordinates"""
        all_words = {}
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                all_words[page_num] = self._extract_page_words(page, page_num)
                self.logger.info(f"Extracted {len(all_words[page_num])} words from page {page_num}")
        return all_words
    
    def _extract_page_words(self, page, page_num: int) -> List[Word]:
        """Extract words from single page"""
        words = []
        for word_obj in page.extract_words():
            words.append(Word(
                text=word_obj['text'],
                x=word_obj['x0'],
                y=word_obj['top'],
                page=page_num
            ))
        return words
    
    def get_page_dimensions(self, page_num: int) -> Tuple[float, float]:
        """Get (width, height) of page"""
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_num]
            return (page.width, page.height)
    
    def get_separator_lines(self, page_num: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get min and max Y values of header/footer separator lines.
        Uses PyMuPDF to detect full-width horizontal lines (>400pt wide)
        Returns: (min_y, max_y) or (None, None) if not found
        """
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]
        drawings = page.get_drawings()
        
        long_lines_y = []
        
        for draw in drawings:
            if draw['type'] == 's':  # Stroke (line)
                items = draw['items']
                if items and items[0][0] == 'l':  # Line operator
                    p1, p2 = items[0][1], items[0][2]
                    
                    # Check if horizontal (same Y within 0.5pt)
                    if abs(p1.y - p2.y) < 0.5:
                        width = abs(p2.x - p1.x)
                        
                        # Long lines: >400pt wide
                        if width > 400:
                            long_lines_y.append(p1.y)
        
        doc.close()
        
        if not long_lines_y:
            self.logger.warning(f"No separator lines found on page {page_num}")
            return None, None
        
        min_y = min(long_lines_y)
        max_y = max(long_lines_y)
        self.logger.debug(f"Page {page_num}: separator lines at Y={min_y:.1f}, {max_y:.1f}")
        
        return min_y, max_y
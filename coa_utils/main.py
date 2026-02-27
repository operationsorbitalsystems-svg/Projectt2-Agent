import json
from pathlib import Path
from datetime import datetime
from .pdf_handler import PDFHandler
from .processor import (
    create_sentences, build_levels_array, assign_levels,
    build_hierarchy, flatten_hierarchy, hierarchy_to_dict, count_nodes
)
from .models import COAOutput, Metadata
from utils.logger import setup_logger
from .config import PDF_INPUT_PATH, OUTPUT_DIR

logger = setup_logger(__name__)

def parse_coa(pdf_path: str, output_dir: str = OUTPUT_DIR, debug: bool = False):
    """Main pipeline"""
    logger.info(f"Starting COA parsing: {pdf_path}")
    
    # Extract words from PDF
    handler = PDFHandler(pdf_path)
    all_words_by_page = handler.extract_all_words()
    
    # Accumulate sentences across all pages
    global_sentences = []
    
    for page_num in sorted(all_words_by_page.keys()):
        page_words = all_words_by_page[page_num]
        logger.info(f"Processing page {page_num}")
        
        # Phase 1: Get separator lines using PyMuPDF
        min_y, max_y = handler.get_separator_lines(page_num)
        
        if min_y is None:
            logger.warning(f"No separators found on page {page_num}, using all words")
            content_words = page_words
        else:
            # Filter words to content zone
            content_words = [w for w in page_words if min_y <= w.y <= max_y]
        
        # Phase 2: Create sentences
        page_sentences = create_sentences(content_words)
        global_sentences.extend(page_sentences)
    
    logger.info(f"Total sentences across all pages: {len(global_sentences)}")
    # Phase 3: Build levels
    levels = build_levels_array(global_sentences)
    
    
    # Phase 4: Assign levels
    global_sentences = assign_levels(global_sentences, levels)
    
    # Phase 5: Build hierarchy
    hierarchy_root = build_hierarchy(global_sentences)
    
    
    # Generate output
    total_groups = count_nodes(hierarchy_root, leaf_only=False)
    total_ledgers = count_nodes(hierarchy_root, leaf_only=True)
    
    metadata = Metadata(
        source_pdf=pdf_path,
        parsed_at=datetime.now(),
        total_pages=len(all_words_by_page),
        total_groups=total_groups,
        total_ledgers=total_ledgers,
        levels_discovered=len(levels)
    )
    
    output = COAOutput(
        metadata=metadata,
        hierarchy=hierarchy_to_dict(hierarchy_root),
        flat_list=flatten_hierarchy(hierarchy_root),
        debug_sentences=[s.dict() for s in global_sentences] if debug else None,
        debug_levels=[l.dict() for l in levels] if debug else None,
        
    )
    
# # Save
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     output_path = Path(output_dir) / "coa_output.json"
#     with open(output_path, 'w') as f:
#         json.dump(output.model_dump(mode='json'), f, indent=2, default=str)
    
#     logger.info(f"Output saved to {output_path}")
    return output

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Parse Tally COA from PDF")
#     parser.add_argument("pdf_path", nargs="?", default=PDF_INPUT_PATH, help="Path to PDF")
#     parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
#     parser.add_argument("--debug", action="store_true", help="Include debug info")
    
#     args = parser.parse_args()
#     parse_coa(args.pdf_path, args.output, args.debug)
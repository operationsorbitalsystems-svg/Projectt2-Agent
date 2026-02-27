from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime

class Word(BaseModel):
    text: str
    x: float
    y: float
    page: int

class Sentence(BaseModel):
    text: str
    x: float
    y: float
    page: int
    level: Optional[int] = None

class Level(BaseModel):
    x_min: float
    x_max: float
    level: int

class HierarchyNode(BaseModel):
    name: str
    level: int
    children: List['HierarchyNode'] = []

    class Config:
        arbitrary_types_allowed = True

class Metadata(BaseModel):
    source_pdf: str
    parsed_at: datetime
    total_pages: int
    total_groups: int
    total_ledgers: int
    levels_discovered: int

class COAOutput(BaseModel):
    metadata: Metadata
    hierarchy: Dict[str, Any]
    flat_list: List[str]
    debug_sentences: Optional[List[Dict]] = None
    debug_levels: Optional[List[Dict]] = None
    sentence_array : List[Sentence] = []
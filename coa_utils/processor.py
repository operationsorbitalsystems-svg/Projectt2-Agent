from typing import List, Tuple, Dict
from .models import Word, Sentence, Level
from .config import X_VARIATION, Y_VARIATION
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ============ PHASE 1: Find Boundaries ============
# Handled by PDFHandler.get_separator_lines() using PyMuPDF

# ============ PHASE 2: Group Words by Y ============

def create_sentences(words: List[Word]) -> List[Sentence]:
    """Group words at same Y into sentences"""
    # Group by Y-coordinate
    y_groups = {}
    for word in words:
        y_rounded = round(word.y / Y_VARIATION) * Y_VARIATION
        if y_rounded not in y_groups:
            y_groups[y_rounded] = []
        y_groups[y_rounded].append(word)
    
    sentences = []
    for y_coord in sorted(y_groups.keys()):
        words_at_y = y_groups[y_coord]
        # Sort by X (left to right)
        words_at_y.sort(key=lambda w: w.x)
        
        # Combine text
        text = ' '.join([w.text for w in words_at_y])
        page = words_at_y[0].page
        x_first = words_at_y[0].x
        
        sentences.append(Sentence(text=text, x=x_first, y=y_coord, page=page))
    
    logger.info(f"Created {len(sentences)} sentences from {len(words)} words")
    return sentences

# ============ PHASE 3: Build Levels Array ============

def build_levels_array(sentences: List[Sentence]) -> List[Level]:
    """Cluster X-coordinates into hierarchy levels"""
    # Collect unique X coordinates
    x_coords = sorted(set(s.x for s in sentences))
    logger.info(f"Unique X-coordinates: {x_coords}")
    
    # Greedy clustering
    levels = []
    for x in x_coords:
        found = False
        for level in levels:
            if abs(x - level['representative_x']) < X_VARIATION:
                # Add to existing level
                level['x_min'] = min(level['x_min'], x)
                level['x_max'] = max(level['x_max'], x)
                found = True
                break
        
        if not found:
            # New level
            levels.append({
                'representative_x': x,
                'x_min': x - X_VARIATION,
                'x_max': x + X_VARIATION,
                'coords': [x]
            })
    
    # Sort by X and assign level numbers
    levels.sort(key=lambda l: l['x_min'])
    result = []
    for idx, level in enumerate(levels):
        result.append(Level(
            x_min=level['x_min'],
            x_max=level['x_max'],
            level=idx
        ))
        logger.debug(f"Level {idx}: X range [{level['x_min']}, {level['x_max']}]")
    
    logger.info(f"Built {len(result)} levels")
    return result

# ============ PHASE 4: Assign Levels ============

def assign_levels(sentences: List[Sentence], levels: List[Level]) -> List[Sentence]:
    """Assign level to each sentence"""
    for sentence in sentences:
        for level in levels:
            if level.x_min <= sentence.x <= level.x_max:
                sentence.level = level.level
                break
        
        if sentence.level is None:
            logger.warning(f"No level found for sentence at X={sentence.x}: {sentence.text}")
    
    logger.info(f"Assigned levels to {len(sentences)} sentences")
    return sentences

# ============ PHASE 5: Build Hierarchy ============

def build_hierarchy(sentences: List[Sentence]) -> List[Dict]:
    """Stack-based hierarchy construction"""
    root = {'name': 'ROOT', 'level': -1, 'children': []}
    stack = [root]

    for sentence in sentences:
        node = {
            'name': sentence.text,
            'level': sentence.level,
            'children': []
        }

        # Pop stack until we find correct parent
        while stack[-1]['level'] >= node['level']:
            stack.pop()

        # Add as child
        parent = stack[-1]
        parent['children'].append(node)
        stack.append(node)

        # logger.debug(f"Added '{sentence.text}' at level {sentence.level}")

    logger.info(f"Built hierarchy with {len(root['children'])} root children")
    return root['children']

def flatten_hierarchy(root: List[Dict]) -> List[str]:
    """Extract all leaf nodes (ledgers) in order"""
    ledgers = []

    def traverse(node):
        if not node['children']:
            # Leaf node
            ledgers.append(node['name'])
        else:
            for child in node['children']:
                traverse(child)

    for child in root:
        traverse(child)

    return ledgers

def hierarchy_to_dict(root: List[Dict]) -> Dict:
    """Convert hierarchy to nested dict (remove 'level' key for output)"""
    result = {}

    def convert_node(node):
        if not node['children']:
            return []  # Leaf node

        converted = {}
        for child in node['children']:
            child_name = child['name']
            child_result = convert_node(child)

            # Handle duplicate names by converting to list
            if child_name in converted:
                # First occurrence: convert to list
                if not isinstance(converted[child_name], list):
                    converted[child_name] = [converted[child_name]]
                # Append new occurrence
                converted[child_name].append(child_result)
            else:
                converted[child_name] = child_result

        return converted

    for child in root:
        child_name = child['name']
        child_result = convert_node(child)

        # Handle duplicates at root level
        if child_name in result:
            if not isinstance(result[child_name], list):
                result[child_name] = [result[child_name]]
            result[child_name].append(child_result)
        else:
            result[child_name] = child_result

    return result

def count_nodes(root: List[Dict], leaf_only: bool = False) -> int:
    """Count nodes in hierarchy"""
    count = 0

    def traverse(node):
        nonlocal count
        if not node['children']:
            if leaf_only:
                count += 1
        else:
            if not leaf_only:
                count += 1
            for child in node['children']:
                traverse(child)

    for child in root:
        traverse(child)

    return count
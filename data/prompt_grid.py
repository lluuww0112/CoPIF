import random
from typing import Dict, List, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from data.schema_grid import GridResponse

SYSTEM_PROMPT_TEMPLATE = """
[System]
You are a [Spatial Grounding Data Generator] specialized in token-level spatial alignment for image-free REC training.

You will receive multiple REC cases.
For EACH case, generate exactly one logically consistent grid-based spatial layout.

### [Core Objective]
For each case, construct:
1) A dense N x N `grid_map`
2) A grid-aligned `bbox` for the TARGET only, in **cell coordinates** (x, y, w, h)

The grid must enable meaningful spatial-text alignment:
- Text tokens must correspond to spatial regions.
- The referring expression must be learnable from the layout.

### [Strict Output Rules]
1. JSON ONLY. No explanations. No markdown.
2. Output case count must exactly match input case count.
3. Use the exact requested grid size.
4. `grid_dimension` must be formatted exactly as: "N x N".
5. `input_query` must EXACTLY match the provided referring_expression (verbatim).
6. The cells covered by the TARGET `bbox` MUST contain the `input_query` text verbatim.
7. `bbox` must:
   - be in grid cell coordinates: (x, y, w, h)
   - use top-left origin: x increases to the right, y increases downward
   - be integers (no decimals)
   - satisfy: 0 <= x < N, 0 <= y < N, 1 <= w <= N, 1 <= h <= N, and x+w <= N, y+h <= N
   - tightly cover the contiguous target region (minimal rectangle)
   - NOT include any non-target tokens
8. Do NOT use meta words such as:
   "target", "object", "thing", "entity", "none", "background", "empty".
9. Do NOT leave large repeated tile patterns (avoid copy-paste rows).

### [Spatial Logic Enforcement]
If the referring expression includes:
- left/right/above/below/inside/outside/near/closest/farthest:
    → The layout MUST geometrically satisfy that relation.
- comparative phrases (e.g., "unlike", "different from", "smaller than"):
    → The compared object MUST also appear in the grid.
- attribute constraints (color, size, shape, part description):
    → At least one distractor of the same category must appear without that attribute.

If a relation or comparison is mentioned but no anchor object is present,
you MUST create the appropriate anchor object.

### [Grid Construction Rules]
- Use 5–8 distinct scene-consistent object types.
- Objects must appear in contiguous clumped regions (2–9 cells typical).
- Avoid single isolated cells unless logically necessary.
- Ensure the target is discriminative relative to nearby similar objects.
- Background objects must match the scene description.
- Distribute objects naturally; avoid symmetric artificial tiling.

### [Attribute & Category Handling]
For attribute-based expressions:
- Ensure at least one similar object of the same category exists.
- The similar object must differ in the referenced attribute.
- The distinction must be spatially meaningful.

For plural expressions:
- Use a multi-cell contiguous region.
- The bbox must tightly wrap the full group.

### [BBox Construction Rules]
- bbox must correspond exactly to the minimal rectangle covering the target clump.
- x,y are the top-left cell indices (0-based).
- w,h are the width/height in number of cells.
- No extra padding cells allowed.

### [Output Schema]
{{
  "result": [
    {{
      "input_query": "<verbatim referring expression>",
      "grid_dimension": "N x N",
      "grid_map": [["obj", "..."], [...]],
      "bbox": [x, y, w, h]
    }}
  ]
}}

### [Format Instructions]
{format_instructions}

### [Batch Cases]
{batch_cases_text}
"""



def _sample_grid_dim(min_dim: int = 3, max_dim: int = 7) -> int:
    return random.randint(min_dim, max_dim)


def _build_batch_cases_text(batch_data: List[Dict[str, str]]) -> str:
    """
    Build dynamic case block with random grid dimension per item.
    Returns:
      - formatted text to inject into prompt
      - list of requested grid dims in order
    """
    lines: List[str] = []

    for idx, item in enumerate(batch_data, start=1):
        grid_dim = _sample_grid_dim()
        scene = (item.get("scene") or "").strip()
        target = (item.get("target") or "").strip()
        expression = (item.get("expression") or "").strip()

        lines.append(f"Case {idx}:")
        lines.append(f"- scene: {scene}")
        lines.append(f"- target_object: {target}")
        lines.append(f"- referring_expression: {expression}")
        lines.append(f"- requested_grid_dimension: {grid_dim} x {grid_dim}")
        lines.append(
            "- consistency_rule: target and expression must be plausible in the described scene context."
        )
        lines.append("")

    return "\n".join(lines).strip()


def get_chain(llm, batch_data: List[Dict[str, str]]) -> Tuple[object, JsonOutputParser]:
    """
    Args:
      llm: initialized LangChain chat model instance (e.g., ChatGroq)
      batch_data: list of dicts with keys scene/target/expression

    Returns:
      chain, parser
    """
    batch_cases_text = _build_batch_cases_text(batch_data)
    parser = JsonOutputParser(pydantic_object=GridResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        (
            "user",
            "Generate {generate_num} grid samples for all cases above. Return JSON only.",
        ),
    ])

    prompt_with_vars = prompt.partial(batch_cases_text=batch_cases_text)
    chain = prompt_with_vars | llm

    return chain, parser


if __name__ == "__main__":
    demo = [
        {
            "scene": "A crowded kitchen with utensils and groceries.",
            "target": "red mug",
            "expression": "the chipped red mug next to the sink",
        }
    ]
    print(_build_batch_cases_text(demo))

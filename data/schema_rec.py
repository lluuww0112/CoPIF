from pydantic import BaseModel, Field
from typing import List



class RECItem(BaseModel):
    """Represents a single Referring Expression Comprehension data entry."""
    scene: str = Field(description="Detailed description of the scene (context).")
    target_object: str = Field(description="The specific object being referred to.")
    referring_expressions: List[str] = Field(description="A list of 3 distinct, natural referring expressions for the target.")
    category_mix: List[str] = Field(description="List of categories used (e.g., Attribute, Spatial, Action, etc.).")

class RECResponse(BaseModel):
    """Container for the list of generated REC items."""
    result: List[RECItem] = Field(description="List of generated REC data items.")
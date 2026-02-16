import math

from pydantic import BaseModel, Field, model_validator
from typing import List


class GridItem(BaseModel):
    input_query: str = Field(description="The full referring expression text (verbatim)")
    grid_dimension: str = Field(description="Grid size, e.g., '10 x 10'")
    grid_map: List[List[str]] = Field(description="2D grid map containing object names")
    bbox: List[float] = Field(
        min_length=4,
        max_length=4,
        description="Target bounding box [x, y, w, h] in grid cell coordinates (0-based, top-left origin)",
    )

    @model_validator(mode="after")
    def validate_item(self):
        rows = len(self.grid_map)
        cols = len(self.grid_map[0]) if rows > 0 else 0
        dim_str = f"{rows} x {cols}"

        # 단순히 포맷을 맞추기 위해 검증 로직 추가 (선택사항)
        if self.grid_dimension != dim_str:
            # LLM이 실수를 할 경우를 대비해 보정하거나 경고를 띄울 수 있음
            pass

        if rows == 0 or cols == 0 or any(len(r) != cols for r in self.grid_map):
            raise ValueError("grid_map must be a non-empty rectangular 2D list")

        # Accept both:
        # - new format: [x, y, w, h] in cell coords
        # - legacy format: [xmin, ymin, xmax, ymax] normalized in [0,1]
        try:
            raw = [float(v) for v in self.bbox]
        except Exception as e:
            raise ValueError(f"Invalid bbox values: {self.bbox}") from e

        if (
            all(0.0 <= v <= 1.0 for v in raw)
            and raw[2] > raw[0]
            and raw[3] > raw[1]
            and (any(abs(v - round(v)) > 1e-6 for v in raw) or raw[2] < 1.0 or raw[3] < 1.0)
        ):
            xmin, ymin, xmax, ymax = raw
            xmin = min(max(xmin, 0.0), 1.0)
            ymin = min(max(ymin, 0.0), 1.0)
            xmax = min(max(xmax, 0.0), 1.0)
            ymax = min(max(ymax, 0.0), 1.0)

            # Convert to cell coordinates.
            x0 = int(xmin * cols)
            y0 = int(ymin * rows)
            x1 = int(math.ceil(xmax * cols) - 1)
            y1 = int(math.ceil(ymax * rows) - 1)

            x0 = max(0, min(cols - 1, x0))
            y0 = max(0, min(rows - 1, y0))
            x1 = max(0, min(cols - 1, x1))
            y1 = max(0, min(rows - 1, y1))

            if x1 < x0 or y1 < y0:
                x, y, w, h = x0, y0, 1, 1
            else:
                x, y, w, h = x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)
        else:
            x, y, w, h = (int(round(v)) for v in raw)

        if x < 0 or y < 0:
            raise ValueError("bbox x,y must be >= 0")
        if w <= 0 or h <= 0:
            raise ValueError("bbox w,h must be >= 1")
        if x + w > cols or y + h > rows:
            raise ValueError("bbox must lie within grid_map bounds")

        # Store as ints (but the field type allows it).
        self.bbox = [int(x), int(y), int(w), int(h)]
        return self


class GridResponse(BaseModel):
    result: List[GridItem] = Field(description="Batch of generated grid samples")

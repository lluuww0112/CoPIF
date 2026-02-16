import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_groq import ChatGroq
from tqdm import tqdm

from data.prompt_grid import get_chain

DATA_DIR = Path(__file__).resolve().parent
REC_INPUT_DIR = DATA_DIR / "res" / "rec"
GRID_OUTPUT_DIR = DATA_DIR / "res" / "grid"


def save_result(data, model_name, total_count):
    """Saves the accumulated generated data to data/res/grid directory."""
    GRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(":", "-").replace("/", "_")
    filename = f"grid_data_{safe_model_name}_n{total_count}_{timestamp}.json"
    filepath = GRID_OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n[Success] All data saved to: {filepath}")


def sanitize_json(text):
    """
    Extract pure JSON from LLM output.
    Removes Markdown code fences and finds the outermost JSON block.
    """
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = text.rfind("}")
        if last_brace != -1:
            return text[first_brace : last_brace + 1]

    if first_bracket != -1:
        last_bracket = text.rfind("]")
        if last_bracket != -1:
            return text[first_bracket : last_bracket + 1]

    return text


def _message_to_text(message) -> str:
    """
    Normalize a LangChain message to plain text.
    Chat model responses can be:
      - str
      - AIMessage(content=str)
      - AIMessage(content=[{"type":"text","text":"..."} , ...])
    """
    if message is None:
        return ""
    if isinstance(message, str):
        return message

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    parts.append(part["text"])
        return "\n".join(p for p in parts if p).strip()

    return str(content) if content is not None else ""


def _extract_rec_items(json_data):
    """Flatten REC json into scene/target/expression tuples."""
    extracted = []

    if isinstance(json_data, dict):
        candidate = json_data.get("result", [])
        if isinstance(candidate, list):
            json_data = candidate
        else:
            json_data = []

    if not isinstance(json_data, list):
        return extracted

    for item in json_data:
        if not isinstance(item, dict):
            continue

        scene = item.get("scene", "")
        target = item.get("target_object", item.get("target", ""))
        expressions = item.get("referring_expressions", item.get("expression", []))

        if not scene or not target:
            continue

        if isinstance(expressions, list):
            for expr in expressions:
                if isinstance(expr, str) and expr.strip():
                    extracted.append(
                        {
                            "scene": scene,
                            "target": target,
                            "expression": expr.strip(),
                        }
                    )
        elif isinstance(expressions, str) and expressions.strip():
            extracted.append(
                {
                    "scene": scene,
                    "target": target,
                    "expression": expressions.strip(),
                }
            )

    return extracted


def get_rec_data(path=None, shuffle=True, seed=None):
    """Load REC-style source data from json files."""
    input_path = Path(path).expanduser().resolve() if path else REC_INPUT_DIR

    if not input_path.exists():
        print(f"[Error] Input directory not found: {input_path}")
        return []

    data_list = []
    file_list = sorted(os.listdir(input_path))

    print(f"[*] Loading REC data from {input_path}...")
    for file_name in file_list:
        if not file_name.endswith(".json"):
            continue

        # Prefer REC outputs and skip previously generated grid outputs.
        if file_name.startswith("grid_data_"):
            continue

        file_path = input_path / file_name
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data_list.extend(_extract_rec_items(raw))
        except Exception as e:
            print(f"[Warning] Failed to load {file_name}: {e}")

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(data_list)

    return data_list


def _parse_grid_items(cleaned_json_text, output_parser):
    """Parse JSON text and normalize to a list of grid items."""
    try:
        parsed_data = output_parser.parse(cleaned_json_text)
    except Exception:
        parsed_data = json.loads(cleaned_json_text)

    if isinstance(parsed_data, dict):
        if isinstance(parsed_data.get("result"), list):
            return parsed_data["result"]
        if all(k in parsed_data for k in ("input_query", "grid_dimension", "grid_map", "bbox")):
            return [parsed_data]
    elif hasattr(parsed_data, "result"):
        result = getattr(parsed_data, "result", None)
        if isinstance(result, list):
            return [item.model_dump() if hasattr(item, "model_dump") else item for item in result]
    elif isinstance(parsed_data, list):
        return parsed_data

    return []


def _normalize_model(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _enforce_target_label_verbatim(grid_items, batch_items):
    """
    Ensure the output uses the exact referring expression (verbatim) from the source REC batch,
    and strip extra fields so each item only contains: input_query, grid_dimension, grid_map, bbox.
    """
    if not isinstance(grid_items, list):
        return grid_items

    batch_items = batch_items or []
    expressions_in_order = []
    for b in batch_items:
        if not isinstance(b, dict):
            continue
        expressions_in_order.append((b.get("expression") or "").strip())

    remaining_by_value = {}
    for expr in expressions_in_order:
        if not expr:
            continue
        remaining_by_value[expr] = remaining_by_value.get(expr, 0) + 1

    def pick_expression(item_idx: int, item_dict: dict) -> str:
        candidate = item_dict.get("input_query")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate and remaining_by_value.get(candidate, 0) > 0:
                remaining_by_value[candidate] -= 1
                return candidate
        if 0 <= item_idx < len(expressions_in_order):
            fallback = expressions_in_order[item_idx]
            if fallback and remaining_by_value.get(fallback, 0) > 0:
                remaining_by_value[fallback] -= 1
            return fallback
        return ""

    normalized_items = []
    for idx, item in enumerate(grid_items):
        item_dict = _normalize_model(item)
        if not isinstance(item_dict, dict):
            continue
        expression = pick_expression(idx, item_dict)

        if expression:
            item_dict["input_query"] = expression

        # Target text in grid_map must exactly equal the referring expression (input_query).
        target_text = expression or ""

        if "bbox" not in item_dict:
            annotations = item_dict.get("annotations")
            if isinstance(annotations, list):
                for ann in annotations:
                    ann_dict = _normalize_model(ann)
                    if not isinstance(ann_dict, dict):
                        continue
                    ann_type = ann_dict.get("type")
                    ann_type_norm = ann_type.lower() if isinstance(ann_type, str) else str(ann_type).lower()
                    if ann_type_norm == "target" and isinstance(ann_dict.get("bbox"), list):
                        item_dict["bbox"] = ann_dict["bbox"]
                        break

        grid_map = item_dict.get("grid_map", [])
        bbox = item_dict.get("bbox", [])
        if (
            target_text
            and isinstance(grid_map, list)
            and grid_map
            and all(isinstance(r, list) for r in grid_map)
            and isinstance(bbox, list)
            and len(bbox) == 4
        ):
            rows = len(grid_map)
            cols = len(grid_map[0]) if rows > 0 else 0
            if cols > 0 and all(len(r) == cols for r in grid_map):
                def _as_floats(vs):
                    try:
                        return [float(v) for v in vs]
                    except Exception:
                        return None

                def _is_legacy_normalized_xyxy(vs):
                    # Legacy format: [xmin, ymin, xmax, ymax] in [0,1].
                    if vs is None or len(vs) != 4:
                        return False
                    xmin, ymin, xmax, ymax = vs
                    if not all(0.0 <= v <= 1.0 for v in vs):
                        return False
                    # Width/height must be > 0 in normalized space.
                    if not (xmax > xmin and ymax > ymin):
                        return False
                    # New format uses integer (x,y,w,h) with w,h >= 1, so
                    # fractional values inside [0,1] are a strong signal of legacy.
                    return any(abs(v - round(v)) > 1e-6 for v in vs) or (xmax < 1.0 or ymax < 1.0)

                bbox_f = _as_floats(bbox)
                x0 = y0 = w = h = None

                if _is_legacy_normalized_xyxy(bbox_f):
                    xmin, ymin, xmax, ymax = bbox_f
                    xmin = min(max(xmin, 0.0), 1.0)
                    ymin = min(max(ymin, 0.0), 1.0)
                    xmax = min(max(xmax, 0.0), 1.0)
                    ymax = min(max(ymax, 0.0), 1.0)

                    x0 = int(math.floor(xmin * cols))
                    x1 = int(math.ceil(xmax * cols) - 1)
                    y0 = int(math.floor(ymin * rows))
                    y1 = int(math.ceil(ymax * rows) - 1)

                    x0 = max(0, min(cols - 1, x0))
                    x1 = max(0, min(cols - 1, x1))
                    y0 = max(0, min(rows - 1, y0))
                    y1 = max(0, min(rows - 1, y1))

                    if x1 < x0 or y1 < y0:
                        cx = int(((xmin + xmax) / 2.0) * cols)
                        cy = int(((ymin + ymax) / 2.0) * rows)
                        cx = max(0, min(cols - 1, cx))
                        cy = max(0, min(rows - 1, cy))
                        x0 = cx
                        y0 = cy
                        w = h = 1
                    else:
                        w = (x1 - x0) + 1
                        h = (y1 - y0) + 1
                elif bbox_f is not None:
                    # New format: [x, y, w, h] in grid cell coordinates.
                    x0, y0, w, h = (int(round(v)) for v in bbox_f)
                    if w <= 0:
                        w = 1
                    if h <= 0:
                        h = 1
                    x0 = max(0, min(cols - 1, x0))
                    y0 = max(0, min(rows - 1, y0))
                    w = max(1, min(cols - x0, w))
                    h = max(1, min(rows - y0, h))

                if x0 is not None and y0 is not None and w is not None and h is not None:
                    x1 = min(cols - 1, x0 + w - 1)
                    y1 = min(rows - 1, y0 + h - 1)

                    grid_map = [list(r) for r in grid_map]
                    for yy in range(y0, y1 + 1):
                        for xx in range(x0, x1 + 1):
                            grid_map[yy][xx] = target_text
                    item_dict["grid_map"] = grid_map

                    # Derive bbox from the actual target_text positions to ensure consistency.
                    xs = []
                    ys = []
                    for yy, row in enumerate(grid_map):
                        for xx, cell in enumerate(row):
                            if cell == target_text:
                                xs.append(xx)
                                ys.append(yy)
                    if xs and ys:
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        item_dict["bbox"] = [
                            int(min_x),
                            int(min_y),
                            int(max_x - min_x + 1),
                            int(max_y - min_y + 1),
                        ]
        elif (
            target_text
            and isinstance(grid_map, list)
            and grid_map
            and all(isinstance(r, list) for r in grid_map)
        ):
            rows = len(grid_map)
            cols = len(grid_map[0]) if rows > 0 else 0
            if cols > 0 and all(len(r) == cols for r in grid_map):
                xs = []
                ys = []
                for yy, row in enumerate(grid_map):
                    for xx, cell in enumerate(row):
                        if cell == target_text:
                            xs.append(xx)
                            ys.append(yy)
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    item_dict["bbox"] = [
                        int(min_x),
                        int(min_y),
                        int(max_x - min_x + 1),
                        int(max_y - min_y + 1),
                    ]

        normalized_items.append(
            {
                "input_query": item_dict.get("input_query", ""),
                "grid_dimension": item_dict.get("grid_dimension", ""),
                "grid_map": item_dict.get("grid_map", []),
                "bbox": item_dict.get("bbox", []),
            }
        )

    return normalized_items


def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Grid Dataset Generator using Groq (Batch Processing)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Groq model name")
    parser.add_argument("--input_dir", type=str, default=str(REC_INPUT_DIR), help="REC input directory path")
    parser.add_argument("--total_num", type=int, default=10, help="Total number of items to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of items per LLM request")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max output tokens per response")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input shuffling")
    parser.add_argument("--api_key", type=str, default=None, help="Groq API key (or use GROQ_API_KEY env var)")
    parser.add_argument("--debug", action="store_true", help="Print extra debug info on failures")

    args = parser.parse_args()

    # 2. Load Source REC Data
    source_data = get_rec_data(args.input_dir, shuffle=True, seed=args.seed)
    if not source_data:
        print("[Error] No valid REC source data found.")
        return

    if args.total_num > 0:
        source_data = source_data[: min(args.total_num, len(source_data))]

    total_target = len(source_data)
    if total_target == 0:
        print("[Error] Nothing to generate after applying total_num.")
        return

    print("[*] Initializing Generator...")
    print(f"    - Model       : {args.model}")
    print(f"    - Total Target: {total_target}")
    print(f"    - Batch Size  : {args.batch_size}")
    print(f"    - Max Tokens  : {args.max_tokens}")

    # 3. Initialize LLM
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Error] GROQ API key is missing. Set GROQ_API_KEY or pass --api_key.")
        return

    try:
        llm = ChatGroq(
            model_name=args.model,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            groq_api_key=api_key,
        )
    except Exception as e:
        print(f"[Error] Failed to connect to Groq: {e}")
        return

    # 4. Batch Processing
    all_results = []
    iterations = (total_target + args.batch_size - 1) // args.batch_size

    print(f"[*] Starting Batch Generation ({iterations} iterations)...")
    start_time = time.time()

    for i in tqdm(range(iterations), desc="Generating Batches"):
        batch_start = i * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_target)
        current_batch = source_data[batch_start:batch_end]
        current_batch_size = len(current_batch)

        chain, output_parser = get_chain(llm, batch_data=current_batch)

        raw_response = ""
        try:
            response_msg = chain.invoke(
                {
                    "generate_num": str(current_batch_size),
                    "format_instructions": output_parser.get_format_instructions(),
                }
            )
            raw_response = _message_to_text(response_msg)
            if not raw_response.strip():
                raise ValueError("Empty response content from LLM")

            cleaned_json_text = sanitize_json(raw_response)
            final_items = _parse_grid_items(cleaned_json_text, output_parser)
            final_items = _enforce_target_label_verbatim(final_items, current_batch)

            if final_items:
                all_results.extend(final_items)
            else:
                print(f"\n[Warning] Batch {i + 1} result is empty.")

            time.sleep(0.5)

        except Exception as e:
            print(f"\n[Warning] Batch {i + 1} failed: {e}")
            print("--- [Debug] Raw Output Preview (First 200 chars) ---")
            print(raw_response[:200] if raw_response else "No response")
            if args.debug:
                try:
                    msg_type = type(response_msg).__name__  # noqa: F821
                except Exception:
                    msg_type = "unknown"
                print(f"--- [Debug] Response Message Type ---\n{msg_type}")
            print("----------------------------------------------------")
            continue

    # 5. Final Save
    total_generated = len(all_results)
    end_time = time.time()

    print("\n[*] Generation Finished.")
    print(f"    - Target: {total_target}")
    print(f"    - Actual: {total_generated}")
    print(f"    - Time  : {end_time - start_time:.2f} seconds")

    if total_generated > 0:
        save_result(all_results, args.model, total_generated)
    else:
        print("[Error] No data was generated. Check debug logs above.")


if __name__ == "__main__":
    main()

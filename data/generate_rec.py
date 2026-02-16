import argparse
import json
import time
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from langchain_ollama import ChatOllama

from data.prompt_rec import get_chain

DATA_DIR = Path(__file__).resolve().parent
REC_OUTPUT_DIR = DATA_DIR / "res" / "rec"


def save_result(data, model_name, total_count):
    """Saves the accumulated generated data to data/res/rec directory."""
    REC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rec_data_{model_name}_n{total_count}_{timestamp}.json"
    filepath = REC_OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\n[Success] All data saved to: {filepath}")


def sanitize_json(text):
    """
    LLM의 출력에서 순수 JSON 부분만 추출합니다.
    Markdown Code Block 제거 및 가장 바깥쪽의 { } 또는 [ ]를 찾습니다.
    """
    # 1. Markdown code block 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()
    
    # 2. 가장 먼저 발견되는 괄호가 '{' 인지 '[' 인지 확인
    # Pydantic 모델(RECResponse)이 객체이므로 '{'를 우선적으로 찾아야 함
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    # '{'가 존재하고, ('['보다 먼저 나오거나 '['가 없을 경우) -> 객체로 간주
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = text.rfind('}')
        if last_brace != -1:
            return text[first_brace : last_brace + 1]
            
    # 위의 경우가 아니고 '['가 존재하면 -> 리스트로 간주
    if first_bracket != -1:
        last_bracket = text.rfind(']')
        if last_bracket != -1:
            return text[first_bracket : last_bracket + 1]

    # 아무것도 못 찾으면 원본 반환 (파서 에러 유도)
    return text


def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="REC Dataset Generator using Ollama (Batch Processing)")
    parser.add_argument("--model", type=str, default="qwen2.5:32b", help="Ollama model name")
    parser.add_argument("--total_num", type=int, default=10, help="Total number of items to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of items per LLM request")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--context_size", type=int, default=4096, help="Context size for LLM")

    args = parser.parse_args()

    print(f"[*] Initializing Generator...")
    print(f"    - Model      : {args.model}")
    print(f"    - Total Target: {args.total_num}")
    print(f"    - Batch Size : {args.batch_size}")
    print(f"    - Context Size : {args.context_size}")

    # 2. Initialize LLM (LLM 객체는 재사용 가능하므로 밖에서 생성)
    try:
        llm = ChatOllama(
            model=args.model,
            temperature=args.temp,
            num_ctx=args.context_size
        )
    except Exception as e:
        print(f"[Error] Failed to connect to Ollama: {e}")
        return

    # 3. Batch Processing Setup
    all_results = []
    # 전체 반복 횟수 계산
    iterations = (args.total_num + args.batch_size - 1) // args.batch_size
    
    print(f"[*] Starting Batch Generation ({iterations} iterations)...")
    start_time = time.time()

    for i in tqdm(range(iterations), desc="Generating Batches"):
        # 현재 배치에서 생성해야 할 실제 개수 계산 (마지막 배치는 batch_size보다 작을 수 있음)
        current_batch_size = min(args.batch_size, args.total_num - len(all_results))
        
        # [핵심] get_chain 호출 시 current_batch_size를 전달
        # 내부적으로 1번부터 N번까지 서로 다른 랜덤 단어가 생성되어 프롬프트에 박힘
        chain, output_parser = get_chain(llm, batch_size=current_batch_size)

        raw_response = "" 
        try:
            # (1) LLM 실행
            response_msg = chain.invoke({
                "generate_num": str(current_batch_size), # 문자열 변환 안전장치
                "format_instructions": output_parser.get_format_instructions()
            })
            raw_response = response_msg.content
            
            # (2) 텍스트 전처리 (JSON 추출)
            cleaned_json_text = sanitize_json(raw_response)
            
            # (3) 파싱 시도
            final_items = []
            try:
                # 3-1. LangChain Parser 시도
                parsed_data = output_parser.parse(cleaned_json_text)
                # Pydantic 객체나 dict로 반환됨
                if isinstance(parsed_data, dict):
                    final_items = parsed_data.get('result', [])
                elif hasattr(parsed_data, 'result'):
                    final_items = parsed_data.result
            except:
                # 3-2. 실패 시 Python json 모듈로 재시도
                json_data = json.loads(cleaned_json_text)
                if isinstance(json_data, dict):
                    final_items = json_data.get('result', [])
                elif isinstance(json_data, list):
                    final_items = json_data

            # (4) 결과 저장
            if final_items:
                all_results.extend(final_items)
            else:
                print(f"\n[Warning] Batch {i+1} result is empty.")

            time.sleep(0.5) # API 부하 방지용 딜레이

        except Exception as e:
            print(f"\n[Warning] Batch {i+1} failed: {e}")
            print(f"--- [Debug] Raw Output Preview (First 200 chars) ---")
            print(raw_response[:200] if raw_response else "No response")
            print("----------------------------------------------------")
            continue

    # 4. Final Save
    total_generated = len(all_results)
    end_time = time.time()
    
    print(f"\n[*] Generation Finished.")
    print(f"    - Target: {args.total_num}")
    print(f"    - Actual: {total_generated}")
    print(f"    - Time  : {end_time - start_time:.2f} seconds")

    if total_generated > 0:
        save_result(all_results, args.model, total_generated)
    else:
        print("[Error] No data was generated. Check the Debug logs above.")

if __name__ == "__main__":
    main()

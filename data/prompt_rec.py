from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from data.schema_rec import RECResponse
from APIs import get_random_words

SYSTEM_PROMPT_TEMPLATE = """
[System]
You are a [Dataset Generation Engine].
You will be provided with a list of 'Ingredient Sets' (numbered 1 to N).
You must generate exactly N REC data samples based on these sets.

### [CRITICAL RULE]
You MUST output the result in a valid JSON object format with a "result" key.
Do NOT output any conversational text.

### [Ingredients Sets]
{random_words_batch_list}

### [Example Output Structure]
(Use this EXACT format)
{{
    "result": [
        {{
            "scene": "A cluttered kitchen counter with various fruits.",
            "target_object": "The yellow banana",
            "referring_expressions": [
                "the curved yellow fruit",
                "the fruit next to the apple",
                "the long yellow object in the bowl"
            ],
            "category_mix": ["Attribute", "Spatial", "Attribute"]
        }},
        {{
            "scene": "A quiet park bench in autumn.",
            "target_object": "The red maple leaf",
            "referring_expressions": [
                "the small red leaf on the seat",
                "the colorful leaf",
                "the leaf fallen from the tree"
            ],
            "category_mix": ["Spatial", "Attribute", "Action"]
        }}
    ]
}}
"""

def get_ingredients_batch_str(batch_size: int, n_words_per_sample: int = 5) -> str:
   """
   배치 크기만큼 서로 다른 랜덤 단어 세트를 생성하여,
   번호가 매겨진 문자열 형태로 반환합니다.
   예:
   1. cat, dog, park
   2. apple, banana, kitchen
   """
   formatted_lines = []
   path = "./APIs/VG/extracted_objects.txt" # 경로 확인 필요
   
   for i in range(batch_size):
      try:
         words_list = get_random_words(path=path, n=n_words_per_sample)
         words_str = ", ".join(words_list)
      except Exception:
         words_str = "random object, unspecified scene"
      
      # "1. word1, word2, word3" 형식으로 저장
      formatted_lines.append(f"{i+1}. {words_str}")
   
   return "\n".join(formatted_lines)


def get_chain(llm, batch_size: int = 5, n_words: int = 5):
   # 1. 배치 재료 생성
   batch_ingredients_str = get_ingredients_batch_str(batch_size, n_words)

   # 2. 파서 설정
   parser = JsonOutputParser(pydantic_object=RECResponse)

   # 3. 템플릿 생성 (format_instructions 제거 - 예시로 대체했으므로 혼동 방지)
   prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_PROMPT_TEMPLATE),
      ("user", "Please generate {generate_num} distinct REC data samples using the ingredients provided above.")
   ])

   # 4. 변수 주입 (batch_ingredients_str만 주입)
   prompt_with_vars = prompt.partial(
      random_words_batch_list=batch_ingredients_str
   )

   # 5. Chain 연결
   chain = prompt_with_vars | llm
   
   return chain, parser
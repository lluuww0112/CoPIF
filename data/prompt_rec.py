from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from data.schema_rec import RECResponse
from APIs import get_random_words

# --- System Prompt 수정: 배치 대응 지침 추가 ---
SYSTEM_PROMPT_TEMPLATE = """
[System]
You are a [Dataset Generation Engine].
You will be provided with a list of 'Ingredient Sets', numbered 1 to N.
You must generate exactly N REC data samples.

### [CRITICAL EXECUTION RULE]
- **One-to-One Mapping**: 
  - For generated sample #1, you MUST use [Ingredient Set #1].
  - For generated sample #2, you MUST use [Ingredient Set #2].
  - And so on.
- Do NOT mix ingredients between sets.

### [Execution Steps for Each Sample]
1. [Scene Setup]: Establish a location/situation based on the assigned ingredient set.
   - **Diversity**: Each sample must have a unique setting.
2. [Target Selection]: Select a specific object based on the ingredients.
   - **Rule**: Describe target using ONLY intrinsic properties (e.g., "A red ball").
3. [Reference Generation]: Generate 5 types of discriminative expressions.

### [5 Key Categories of Referring Expressions]
- Attribute, Spatial, Comparative, Action, Part-of-Whole

### [Ingredients Sets]
The following list provides specific random words for each sample in this batch:
{random_words_batch_list}

### [Output Format]
IMPORTANT: Output ONLY the JSON object (a list of objects). 
No conversational text.
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
   """
   배치 크기에 맞춰 각각 다른 재료를 포함한 프롬프트를 생성하고 Chain을 반환합니다.
   """
   # 1. 배치 사이즈만큼의 서로 다른 재료 목록 생성
   batch_ingredients_str = get_ingredients_batch_str(batch_size, n_words)

   # 2. 파서 설정
   parser = JsonOutputParser(pydantic_object=RECResponse)

   # 3. 템플릿 생성
   prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_PROMPT_TEMPLATE),
      ("user", "Format Instructions:\n{format_instructions}"),
      ("user", "Please generate {generate_num} distinct REC data samples. Use Ingredient Set #1 for the first item, Set #2 for the second, etc.")
   ])

   # 4. 변수 주입 (batch_ingredients_str를 여기에 넣습니다)
   prompt_with_vars = prompt.partial(
      random_words_batch_list=batch_ingredients_str,
      format_instructions=parser.get_format_instructions()
   )

   # 5. Chain 연결
   chain = prompt_with_vars | llm
   
   return chain, parser
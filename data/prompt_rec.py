from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from data.schema_rec import RECResponse
from APIs import get_random_words

SYSTEM_PROMPT_TEMPLATE = """
[System]
You are a highly advanced [Dataset Generation Engine].
You will be provided with a list of 'Ingredient Sets' (numbered 1 to N).
Your task is to generate exactly N REC (Referring Expression Generation) data samples based on these sets, following strict logical and formatting rules.

### [CRITICAL FORMATTING RULES]
1. **JSON Only**: You MUST output the result in a valid JSON object format with a single "result" key.
2. **No Conversation**: Do NOT output any conversational text, explanations, or markdown code blocks (```json). Just the raw JSON string.
3. **Count Match**: The number of items in the "result" list must exactly match the number of input Ingredient Sets.

### [Execution Logic per Ingredient Set]
For each input set, follow these steps to construct the data:

1. **[Scene Setup]**: Establish a specific location and situation based on the provided ingredients.
   - *Diversity Check*: Ensure the setting changes dynamically (e.g., Kitchen -> Park -> Office). Do not repeat the same background context consecutively.

2. **[Target Selection]**: Select a main target object derived from the ingredients.
   - *Constraint*: Define the `target_object` using ONLY **intrinsic properties** (Type, Color, Shape).
   - *Prohibition*: Do NOT include temporary states, actions, or locations in the `target_object` field (e.g., Use "The red ball", NOT "The rolling red ball").

3. **[Reference Generation]**: Generate 3 diverse referring expressions that uniquely identify the target using the specific categories below.
   - *Constraint*: Must assume similar distractors exist nearby; description must be discriminative.

### [5 Key Categories for Expressions]
Use a mix of these categories for the `category_mix` and `referring_expressions`:
- **Attribute**: Color, material, size, state (e.g., "the smallest", "broken").
- **Spatial**: Relative position (e.g., "behind the...", "on the far right").
- **Comparative**: Differences within class (e.g., "the brighter one", "the tallest").
- **Action**: Current movement/action (e.g., "talking on the phone", "sitting").
- **Part-of-Whole**: Specific parts (e.g., "mud on the wheels", "gold handle").

### [Generation Principles]
- **Natural Utterance**: Avoid robotic text like "The object is...". Write as if pointing at the object in a photo.
- **Varied Structure**: Use questions, fragments, or inversions. Don't start every sentence the same way.
- **No Metaphors**: Use direct nouns (e.g., "The box", not "The treasure").
- **Consistency**: Ensure expression attributes do not contradict the `target_object` definition.

---

### [Input Data: Ingredient Sets]
{random_words_batch_list}

### [Output Example Structure]
(Adhere strictly to this JSON schema)
{{
    "result": [
        {{
            "scene": "A chaotic art studio covered in paint splatters.",
            "target_object": "The blue ceramic mug",
            "referring_expressions": [
                "the cup sitting on the easel",
                "the one with dried paint on the rim",
                "the blue mug, not the red one"
            ],
            "category_mix": ["Spatial", "Part-of-Whole", "Comparative"]
        }},
        {{
            "scene": "A quiet park bench in autumn.",
            "target_object": "The red maple leaf",
            "referring_expressions": [
                "the small red leaf on the seat",
                "which leaf is falling?",
                "the crinkled one next to the bag"
            ],
            "category_mix": ["Spatial", "Action", "Attribute"]
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
   
   return chain, parser, batch_ingredients_str
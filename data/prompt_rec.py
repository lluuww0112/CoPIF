from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from data.schema_rec import RECResponse
from APIs import get_random_words


SYSTEM_PROMPT_TEMPLATE = """
[System]
You are a highly advanced [Visual Perception & Grounding Engine].
You will be provided with a list of 'Target Keywords'.
Your task is to generate exactly N REC (Referring Expression Comprehension) data samples based on these keywords.

### [CRITICAL VISUAL CONSTRAINTS - MUST READ]
You must act as a **Camera** or **CCTV**, not a storyteller.
1.  **Strictly Visual Only**: Describe ONLY what can be verified by looking at a static JPEG image.
    * NO Abstract/Emotional terms: (e.g., "kind", "honest", "reliable", "brave", "delicious").
    * NO Non-Visual Senses: (e.g., "loud", "smelly", "hot temperature", "tasty").
    * NO Invisible Relationships: (e.g., "the parson's daughter", "Mavin's favorite", "gift from mom"). Use "girl", "woman", "object" instead.
    * NO Temporal/Causal Inference: (e.g., "waiting for a bus", "wrinkled from laughing", "broken by accident"). Describe only the current state ("standing at the stop", "wrinkled eyes", "broken pieces").
    * NO Dynamic Action in Static Image: (e.g., "flickering light", "shivering"). Use static states ("dim light", "huddled posture").

2.  **Lexical Precision**: Use standard, dictionary-defined nouns.
    * Avoid ambiguous or brand-specific slang unless visually obvious (e.g., NOT "Golden Ritz", USE "Golden Pastry").
    * NOT "Sipper", USE "Tumbler" or "Cup".
    * NOT "Pyre" (unless it's a funeral fire), USE "Pile" or "Stack".

3.  **Referring Expression Validity (REC-SAFE)**: Every referring expression must be a standalone, unambiguous noun phrase.
    * **Always include an explicit head noun** (target category): "the blue ceramic mug", "the striped pencil", "the man in a black jacket".
    * **Never use vague placeholders** that don't specify a category:
      - Forbidden: "the one", "that one", "the one next to ~", "the other one", "it", "this", "that".
      - Bad: "the one next to the cup"
      - Good: "the red apple next to the cup"
    * **Relational/spatial expressions must name both objects**: "the mug next to the silver spoon" (NOT "the one next to the spoon").
    * **Comparatives must state the comparison set explicitly** (with a plural noun or a clear container/group):
      - Bad: "the smallest one", "the darker one", "the one closest to the wall"
      - Good: "the smallest pencil among the pencils", "the darkest apple in the bowl", "the chair closest to the wall"

### [Execution Logic]
For each input keyword:
1. **[Target Definition]**:
   - The provided keyword is the seed.
   - You MUST convert it into a concrete **Noun Phrase** for the `target_object`.
   - Example: Input "Rowdy" -> Target "The rowdy group" (NOT just "Rowdy").
   - Example: Input "Old" -> Target "The old book".
2.  **[Scene Setup]**: Invent a realistic, static visual scene.
3.  **[Distractor Simulation]**: Imagine similar objects nearby to necessitate specific descriptions.
4.  **[Reference Generation]**: Create 3 expressions based on **Pixel-Level Features** (Color, Shape, Texture, Position, Adjacency).
    - Each expression must be independently resolvable without relying on "one/that/it".
    - Prefer describing the target's category + attributes + a visible landmark or comparison set.

### [5 Key Categories for Expressions]
- **Attribute**: Visual properties (e.g., "red", "rusty", "torn", "glossy").
- **Spatial**: X/Y coordinates or relation to landmarks (e.g., "on the left", "closest to the wall").
- **Comparative**: Visual contrast with an explicit comparison set (e.g., "the darker apple in the bowl", "larger than the green apple").
- **State**: Visible physical condition (e.g., "open", "empty", "shattered"). *Replaces abstract 'Action'.*
- **Part-of-Whole**: Visible components (e.g., "with a wooden handle", "missing a button").

### [Formatting Rules]
1.  **JSON Only**: Output a valid JSON object with a single "result" key.
2.  **No Conversation**: Return raw JSON string only.
3.  **Count Match**: Ensure output count equals input count.

### [Input Data: Target Keywords]
{random_words_batch_list}

### [Output Example Structure]
{{
   "result": [
      {{
         "scene": "A wooden table in a sunlit kitchen.",
         "target_object": "The ceramic mug", 
         "referring_expressions": [
            "blue mug with a chipped rim",
            "ceramic mug sitting on the woven coaster",
            "empty ceramic mug, unlike the other mug filled with coffee"
         ],
         "category_mix": ["Attribute", "Spatial", "Comparative"]
      }}
   ]
}}
"""


def get_ingredients_batch_str(batch_size: int) -> str:
   """
   배치 크기만큼 서로 다른 랜덤 단어 세트를 생성하여,
   번호가 매겨진 문자열 형태로 반환합니다.
   예:
   1. cat
   2. apple
   """
   path = "./APIs/VG/extracted_objects.txt" # 경로 확인 필요

   words_list = get_random_words(file_path=path, n=batch_size)
   formatted_lines = [f"{idx+1}. {word}" for idx, word in enumerate(words_list)]


   return "\n".join(formatted_lines)


def get_chain(llm, batch_size: int = 5):
   # 1. 배치 재료 생성
   batch_ingredients_str = get_ingredients_batch_str(batch_size)

   # 2. 파서 설정
   parser = JsonOutputParser(pydantic_object=RECResponse)

   # 3. 템플릿 생성 (format_instructions 제거 - 예시로 대체했으므로 혼동 방지)
   prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_PROMPT_TEMPLATE),
      ("user", "Generate {generate_num} distinct REC data samples using the ingredients provided above.")
   ])

   # 4. 변수 주입 (batch_ingredients_str만 주입)
   prompt_with_vars = prompt.partial(
      random_words_batch_list=batch_ingredients_str
   )

   # 5. Chain 연결
   chain = prompt_with_vars | llm
   
   return chain, parser


if __name__ == "__main__":
   ingredients = get_ingredients_batch_str(batch_size=5)
   print(ingredients)

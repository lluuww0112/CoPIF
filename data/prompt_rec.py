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
1. **Strictly Visual Only**: Describe ONLY what can be verified by looking at a static JPEG image.
   - NO Abstract/Emotional terms (e.g., "kind", "honest", "reliable", "brave", "delicious").
   - NO Non-Visual senses (e.g., "loud", "smelly", "hot temperature", "tasty").
   - NO Invisible relationships (e.g., "the parson's daughter", "Mavin's favorite", "gift from mom").
   - NO Temporal/Causal inference (e.g., "waiting for a bus", "broken by accident").
   - NO Dynamic action in static image (e.g., "flickering", "shivering"). Use visible states only.

2. **Lexical Precision**: Use standard, dictionary-defined nouns.
   - Avoid ambiguous or brand-specific slang unless visually obvious.
   - NOT "Sipper", USE "Tumbler" or "Cup".
   - NOT "Pyre" (unless funeral fire), USE "Pile" or "Stack".

### [ANTI-TEMPLATE / DIVERSITY RULES - ABSOLUTELY CRITICAL]
Small models tend to copy the example. YOU MUST NOT DO THAT.
A) **Example is FORMAT ONLY**: The example below is ONLY to show JSON keys. It is NOT a style guide.
B) **Do NOT imitate example phrasing**:
   - Do NOT reuse the example's sentence patterns, rhythm, or word order.
   - Do NOT repeatedly use the same leading forms like:
     "the one with ...", "the ... on the ...", "..., unlike the ..."
   - Do NOT keep the same 3-expression structure pattern across items.
C) **Hard variety requirement inside each sample (3 expressions)**:
   - Expression #1, #2, #3 MUST start differently (no shared first 1~2 words).
   - Use different syntactic forms across the 3:
     (noun phrase / prepositional phrase / comparative clause / part phrase etc.)
   - Avoid repeating the same adjective pair or the same anchor phrase.
D) **Hard variety requirement across samples**:
   - Do NOT reuse the same scene framing (e.g., "A wooden table in a sunlit kitchen.") repeatedly.
   - Do NOT reuse the same anchor landmarks (wall/table/shelf) across consecutive items.
   - Vary viewpoint words (foreground/background/center/edge/corner) and object relations (next to/behind/overlapping/inside).
E) If you notice you are repeating a pattern, you MUST rewrite before finalizing output.

### [Execution Logic]
For each input keyword:
1. **[Target Definition]**
   - Convert keyword into a concrete **Noun Phrase** for `target_object`.
   - Example: Input "Rowdy" -> Target "The rowdy group".
   - Example: Input "Old" -> Target "The old book".
2. **[Scene Setup]**: Invent a realistic, static visual scene.
3. **[Distractor Simulation]**: Imagine similar objects nearby to necessitate specific descriptions.
4. **[Reference Generation]**: Create 3 expressions based on **Pixel-Level Features** (Color, Shape, Texture, Position, Adjacency).

### [5 Key Categories for Expressions]
- **Attribute**: Visual properties (e.g., "red", "rusty", "torn", "glossy").
- **Spatial**: X/Y coordinates or relation to landmarks (e.g., "on the left", "closest to the wall").
- **Comparative**: Visual contrast (e.g., "the darker one", "larger than the apple").
- **State**: Visible physical condition (e.g., "open", "empty", "shattered").
- **Part-of-Whole**: Visible components (e.g., "with a wooden handle", "missing a button").

### [Formatting Rules]
1. **JSON Only**: Output a valid JSON object with a single "result" key.
2. **No Conversation**: Return raw JSON string only.
3. **Count Match**: Ensure output count equals input count.

### [Input Data: Target Keywords]
{random_words_batch_list}

### [Output Example Structure]  (FORMAT ONLY - DO NOT COPY WORDING OR PATTERNS)
{{
  "result": [
    {{
      "scene": "Example scene text.",
      "target_object": "Example noun phrase.",
      "referring_expressions": ["ex1", "ex2", "ex3"],
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

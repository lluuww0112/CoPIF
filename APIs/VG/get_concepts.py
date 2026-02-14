import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# --- 1. 설정 및 데이터 다운로드 ---
# NLTK 데이터 다운로드 (최초 1회 실행 필요, 이미 되어있다면 주석 처리 가능)
print("NLTK 데이터 다운로드 중...")
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger') # 기본 태거 추가
print("다운로드 완료.")

# 분류기 초기화
lemmatizer = WordNetLemmatizer()

# --- 2. 핵심 로직 함수 ---
def classify_word(word):
    """
    단어가 장소(Place)인지 객체(Object)인지 WordNet을 통해 분류
    """
    lemma = lemmatizer.lemmatize(word.lower())
    synsets = wn.synsets(lemma, pos=wn.NOUN)
    
    if not synsets:
        return "Unknown"

    synset = synsets[0] # 가장 주된 의미 사용
    hypernym_paths = synset.hypernym_paths()
    
    for path in hypernym_paths:
        for hypernym in path:
            if hypernym.name() in ['location.n.01', 'place.n.01', 'geographical_area.n.01']:
                return "Place"
            if hypernym.name() in ['artifact.n.01', 'living_thing.n.01', 'structure.n.01']:
                return "Object"
                
    return "Unknown"

def extract_from_objects_json(file_path):
    print(f"\nProcessing {file_path}...")
    extracted_data = {'Place': set(), 'Object': set()}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        return extracted_data

    # [수정됨] 전체 데이터를 처리하기 위해 슬라이싱([:100]) 제거
    for entry in tqdm(data, desc="Objects JSON 처리 중"): 
        for obj in entry.get('objects', []):
            names = obj.get('names', [])
            for name in names:
                category = classify_word(name)
                if category in ['Place', 'Object']:
                    extracted_data[category].add(name)
                    
    return extracted_data

def extract_from_regions_json(file_path):
    print(f"\nProcessing {file_path}...")
    extracted_data = {'Place': set(), 'Object': set()}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        return extracted_data
        
    # [수정됨] 전체 데이터를 처리하기 위해 슬라이싱([:100]) 제거
    for entry in tqdm(data, desc="Regions JSON 처리 중"): 
        for region in entry.get('regions', []):
            phrase = region.get('phrase', "")
            
            # 문장에서 명사만 추출
            tokens = nltk.word_tokenize(phrase)
            tags = nltk.pos_tag(tokens)
            
            for word, tag in tags:
                if tag.startswith('NN'): # 명사
                    category = classify_word(word)
                    if category in ['Place', 'Object']:
                        extracted_data[category].add(word)
    
    return extracted_data

def save_list_to_txt(data_set, filename):
    """
    집합(Set) 데이터를 텍스트 파일로 저장하는 함수
    """
    print(f"\n파일 저장 중: {filename} ...")
    sorted_list = sorted(list(data_set)) # 가나다순 정렬
    
    with open(filename, 'w', encoding='utf-8') as f:
        for item in sorted_list:
            f.write(f"{item}\n")
    
    print(f"완료! ({len(sorted_list)}개 저장됨)")

# --- 3. 메인 실행 부분 ---

# 실제 파일 경로 (환경에 맞게 수정 필요)
obj_path = './data/objects.json'
reg_path = './data/region_descriptions.json'

# 추출 실행
results_obj = extract_from_objects_json(obj_path)
results_reg = extract_from_regions_json(reg_path)

# 결과 합치기 (Set의 합집합 연산으로 중복 제거)
final_places = results_obj['Place'].union(results_reg['Place'])
final_objects = results_obj['Object'].union(results_reg['Object'])

print("\n" + "="*30)
print("  결과 요약  ")
print("="*30)
print(f"🏠 추출된 장소 (Place): {len(final_places)}개")
print(f"📦 추출된 객체 (Object): {len(final_objects)}개")

# 파일로 저장
save_list_to_txt(final_places, 'extracted_places.txt')
save_list_to_txt(final_objects, 'extracted_objects.txt')

print("\n모든 작업이 완료되었습니다.")
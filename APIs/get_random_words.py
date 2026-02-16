import random
import os

def get_random_words(file_path, n):
    """
    텍스트 파일에서 n개의 랜덤 단어를 추출하는 함수
    (파일은 한 줄에 하나의 단어가 있다고 가정)
    
    Args:
        file_path (str): 파일 경로
        n (int): 추출할 단어의 개수
        
    Returns:
        list: 추출된 단어 리스트
    """
    if not os.path.exists(file_path):
        print("Error: 파일을 찾을 수 없습니다.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 줄 단위로 읽어서 공백/줄바꿈 제거 후 리스트로 변환
            words = [line.strip() for line in file if line.strip()]
        
        # 파일에 있는 단어 수보다 요청한 n이 클 경우 처리
        if n > len(words):
            print(f"주의: 파일의 단어 수({len(words)}개)보다 요청한 개수({n}개)가 많습니다.")
            # 가능한 최대 개수(전체)만 반환하거나, 중복 허용이 필요하면 로직 변경 필요
            return words 
            
        # 중복 없이 n개 추출 (비복원 추출)
        return random.sample(words, n)

    except Exception as e:
        print(f"Error: {e}")
        return []

# --- 사용 예시 ---
# result = get_random_words('./vocabulary.txt', 3)
# print(f"추출된 단어들: {result}")

if __name__ == "__main__":
    path = ""
    results = get_random_words()
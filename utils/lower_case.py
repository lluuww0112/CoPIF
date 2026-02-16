import os

def process_unique_words(file_path):
    try:
        # 1. 파일 읽기 (Read Mode)
        with open(file_path, 'r', encoding='utf-8') as f:
            # 줄 단위로 읽어서 공백 제거 -> 소문자 변환 -> set으로 중복 제거
            unique_words = {line.strip().lower() for line in f if line.strip()}
        
        # 2. 정렬 (알파벳 순)
        sorted_words = sorted(list(unique_words))
        
        # 3. 파일 쓰기 (Write Mode - 원본 덮어쓰기)
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in sorted_words:
                f.write(word + '\n')
        
        print(f"Success: {file_path} 파일이 갱신되었습니다.")
        return sorted_words

    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요.\n({file_path})")
        return []
    except Exception as e:
        print(f"Error: 오류가 발생했습니다. {e}")
        return []


# --- 실행 ---

if __name__ == "__main__":
    path = "./APIs/VG/extracted_objects.txt"
    
    # 함수 실행 (파일 내용이 변경됨)
    result = process_unique_words(path)
    
    print(f"중복 제거 및 저장 완료된 단어 개수: {len(result)}")
    # print(result[:10]) # 변경된 내용 앞 10개 확인
"""
preprocess.py
- 세종 코퍼스 JSON에서 (문장, 형태소, 품사) 추출 및 정제
"""
import json
import re
import pickle

def load_json_corpus(path):
    """
    JSON 파일에서 (문장, [(형태소, 품사)]) 시퀀스 추출

    Args:
        path (str): JSON 파일 경로

    Returns:
        list: (문장, [(형태소, 품사)]) 튜플의 리스트
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        for doc in corpus.get('document', []):
            for sent in doc.get('sentence', []):
                sentence = sent.get('form', '')
                morphemes = sent.get('morpheme') or []
                morphs = [(m['form'], m['label']) for m in morphemes]
                data.append((sentence, morphs))
    return data

def clean_text(text):
    """
    특수문자, 불필요 태그 등 정제

    Args:
        text (str): 입력 문자열

    Returns:
        str: 정제된 문자열
    """
    text = re.sub(r'<[^>]+>', '', text)  # 태그 제거
    text = re.sub(r'[^\w\s가-힣.,!?]', '', text)  # 한글, 영문, 숫자, 공백, 일부 구두점만 남김
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_preprocessed_data(data, output_path):
    """
    전처리된 데이터를 pickle로 저장

    Args:
        data (list): 전처리된 데이터
        output_path (str): 저장할 파일 경로
    """
    with open(output_path, 'wb') as f:
        pickle.dump(data, f) 
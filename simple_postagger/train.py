"""
train.py
- HMM 파라미터(초기/전이/방출 확률) 추정
- 형태소-품사 사전 관련 함수는 dictionary.py로 이동
"""
import pickle
from collections import defaultdict, Counter

def estimate_hmm_parameters(sentences):
    """
    HMM 파라미터(초기, 전이, 방출, 품사 집합) 계산

    Args:
        sentences (list): List of (sentence, [(morph, tag), ...]) tuples.

    Returns:
        tuple: (init_probs, trans_probs, emit_probs, tag_set)
            - init_probs (dict): 초기 확률
            - trans_probs (dict): 전이 확률
            - emit_probs (dict): 방출 확률
            - tag_set (set): 품사 집합
    """
    tag_set = set()
    init_counter = Counter()
    trans_counter = defaultdict(Counter)
    emit_counter = defaultdict(Counter)
    for _, morphs in sentences:
        tags = [tag for _, tag in morphs]
        morphs_only = [morph for morph, _ in morphs]
        if not tags:
            continue
        tag_set.update(tags)
        init_counter[tags[0]] += 1
        for i in range(len(tags)):
            emit_counter[tags[i]][morphs_only[i]] += 1
            if i > 0:
                trans_counter[tags[i-1]][tags[i]] += 1
    # 확률화
    total_init = sum(init_counter.values())
    init_probs = {tag: count/total_init for tag, count in init_counter.items()}
    trans_probs = {tag: {t: c/sum(nexts.values()) for t, c in nexts.items()} for tag, nexts in trans_counter.items()}
    emit_probs = {tag: {m: c/sum(morphs.values()) for m, c in morphs.items()} for tag, morphs in emit_counter.items()}
    return init_probs, trans_probs, emit_probs, tag_set

def save_model(params, path):
    """
    학습된 파라미터를 pickle로 저장

    Args:
        params (tuple): (init_probs, trans_probs, emit_probs, tag_set)
        path (str): 저장할 파일 경로
    """
    with open(path, 'wb') as f:
        pickle.dump(params, f) 
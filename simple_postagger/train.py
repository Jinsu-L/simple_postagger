"""
train.py
- 형태소-품사 사전 구축, HMM 파라미터(초기/전이/방출 확률) 추정
"""
import pickle
from collections import defaultdict, Counter

def build_dictionary(sentences):
    """
    형태소-품사 쌍 빈도 집계
    반환: Counter((morph, tag))
    """
    morph_tag_counter = Counter()
    for _, morphs in sentences:
        for morph, tag in morphs:
            morph_tag_counter[(morph, tag)] += 1
    return morph_tag_counter

def estimate_hmm_parameters(sentences):
    """
    HMM 파라미터(초기, 전이, 방출 확률) 계산
    반환: (init_probs, trans_probs, emit_probs, tag_set)
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
    """
    with open(path, 'wb') as f:
        pickle.dump(params, f)

# 최소 테스트 코드 (실행 예시)
if __name__ == '__main__':
    # 예시: data/NIKL_MP(v1.1)/preprocessed.pkl
    input_path = '../data/NIKL_MP(v1.1)/preprocessed.pkl'
    model_path = '../data/NIKL_MP(v1.1)/hmm_model.pkl'
    # with open(input_path, 'rb') as f:
    #     sentences = pickle.load(f)
    # morph_dict = build_dictionary(sentences)
    # init_p, trans_p, emit_p, tag_set = estimate_hmm_parameters(sentences)
    # save_model((init_p, trans_p, emit_p, tag_set), model_path)
    print('Training module ready.') 
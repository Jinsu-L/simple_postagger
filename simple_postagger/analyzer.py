"""
analyzer.py
- 학습된 모델을 불러와 실제 문장에 대해 형태소 분석 수행 (lattice 기반)
"""
import pickle
from .viterbi import viterbi

def load_model(path):
    """
    학습된 HMM 파라미터 로드

    Args:
        path (str): 저장된 모델 파일 경로

    Returns:
        tuple: (init_probs, trans_probs, emit_probs, tag_set)
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def analyze(sentence, model, method='sentence_lattice'):
    """
    입력 문장에 대해 형태소 분석 수행 및 결과 반환

    Args:
        sentence (str): 입력 문장
        model (tuple): (init_probs, trans_probs, emit_probs, tag_set)
        method (str): 분석 방식 ('lattice', 'beam', 'sentence_lattice')

    Returns:
        list: 분석 결과 (어절, 품사) 또는 품사 시퀀스
    """
    init_p, trans_p, emit_p, tag_set = model
    tag_list = list(tag_set)
    words = sentence.strip().split()
    results = []
    if method == 'beam':
        import itertools
        top_k = 5
        word_candidates = []
        for word in words:
            obs = [word]
            tags, score = viterbi(obs, tag_list, init_p, trans_p, emit_p)
            word_candidates.append([(tags, score)])
        best_score = -float('inf')
        best_seq = None
        for candidate_seq in itertools.product(*word_candidates):
            tag_seq = []
            total_score = 1.0
            prev_tag = None
            for (tags, score) in candidate_seq:
                tag_seq.extend(tags)
                total_score *= score
                if prev_tag is not None:
                    total_score *= trans_p.get(prev_tag, {}).get(tags[0], 1e-8)
                prev_tag = tags[-1]
            if total_score > best_score:
                best_score = total_score
                best_seq = tag_seq
        return best_seq if best_seq else []
    elif method == 'sentence_lattice':
        obs = words
        tags, score = viterbi(obs, tag_list, init_p, trans_p, emit_p)
        return list(zip(obs, tags))
    else:
        for word in words:
            obs = [word]
            tags, score = viterbi(obs, tag_list, init_p, trans_p, emit_p)
            results.append((word, tags[0]))
        return results 
"""
analyzer.py
- 학습된 모델을 불러와 실제 문장에 대해 형태소 분석 수행 (lattice 기반)
"""
import pickle
from .viterbi import viterbi

def load_model(path):
    """
    학습된 HMM 파라미터 로드
    반환: (init_probs, trans_probs, emit_probs, tag_set)
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def analyze(sentence, model, method='lattice'):
    """
    입력 문장에 대해 형태소 분석 수행 및 결과 반환
    method: 'lattice' (기본, 어절별 viterbi), 'beam' (어절간 품사 전이 반영), 'sentence_lattice' (문장 전체 viterbi)
    """
    init_p, trans_p, emit_p, tag_set = model
    tag_list = list(tag_set)
    words = sentence.strip().split()
    results = []
    if method == 'beam':
        # Beam search: 어절별 top-k viterbi 결과 조합 중 전체 score 최대
        import itertools
        top_k = 3
        word_candidates = []
        for word in words:
            # 각 어절을 가능한 모든 형태소(띄어쓰기 없는 경우 그대로)로 보고 viterbi 적용
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
        # 문장 전체를 관측 시퀀스로 보고 viterbi 적용
        obs = words
        tags, score = viterbi(obs, tag_list, init_p, trans_p, emit_p)
        return list(zip(obs, tags))
    else:
        # 기본: 어절별 viterbi 적용
        for word in words:
            obs = [word]
            tags, score = viterbi(obs, tag_list, init_p, trans_p, emit_p)
            results.append((word, tags[0]))
        return results

# 예시 실행 코드
if __name__ == '__main__':
    model = load_model('data/NIKL_MP(v1.1)/hmm_model.pkl')
    sentence = "아버지 가방에 들어가신다"
    result = analyze(sentence, model, method='sentence_lattice')
    print(result) 
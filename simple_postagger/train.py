import pickle
from collections import defaultdict, Counter

def normalize_with_smoothing(counter, smoothing=1e-3, vocab_size=None):
    """
    Additive smoothing을 적용하여 확률화합니다.
    """
    vocab_size = vocab_size or len(counter)
    total = sum(counter.values()) + smoothing * vocab_size
    return {key: (count + smoothing) / total for key, count in counter.items()}

def estimate_hmm_parameters(sentences, smoothing=1e-3, unk_threshold=1):
    """
    HMM 파라미터(초기, 전이, 방출, 품사 집합) 계산

    Args:
        sentences (list): List of (sentence, [(morph, tag), ...]) tuples.
        smoothing (float): Additive smoothing 계수
        unk_threshold (int): UNK 처리할 최소 빈도 이하의 형태소 기준

    Returns:
        tuple: (init_probs, trans_probs, emit_probs, tag_set)
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
            tag = tags[i]
            morph = morphs_only[i]
            emit_counter[tag][morph] += 1
            if i > 0:
                trans_counter[tags[i - 1]][tag] += 1

    # UNK 방출 확률 추가 (희귀 형태소에 대해)
    for tag, morphs in emit_counter.items():
        unk_count = sum(c for m, c in morphs.items() if c <= unk_threshold)
        if unk_count > 0:
            emit_counter[tag]['<UNK>'] += unk_count

    # 확률화 + smoothing
    total_init = sum(init_counter.values())
    init_probs = normalize_with_smoothing(init_counter, smoothing, vocab_size=len(tag_set))

    trans_probs = {
        tag: normalize_with_smoothing(nexts, smoothing, vocab_size=len(tag_set))
        for tag, nexts in trans_counter.items()
    }

    emit_probs = {
        tag: normalize_with_smoothing(morphs, smoothing)
        for tag, morphs in emit_counter.items()
    }

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

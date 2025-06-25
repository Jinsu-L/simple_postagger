import pickle
import math
import re
from simple_postagger.preprocess import clean_text

# UNK 방출 확률 상수
UNK_EMIT_PROB = 1e-8

def is_valid_unk(morph):
    """
    UNK로 허용할 수 있는 형태인지 판단하는 조건 함수.
    커스터마이징 가능.
    예: 조사, 숫자, 특수기호 등은 제외
    """
    # 한 글자이면서 조사나 숫자/특수문자 → 제외
    if len(morph) == 1:
        if re.fullmatch(r'[은는이가을를에의]', morph):  # 대표 조사
            return False
        if re.fullmatch(r'[0-9]|[^\w가-힣]', morph):  # 숫자 또는 특수문자
            return False
    return True  # 허용

def build_lattice(seq, morph_lexicon, unk_min=4, unk_max=5):
    N = len(seq)
    lattice = [[] for _ in range(N+1)]
    for i in range(N):
        has_known = False
        for j in range(i+1, N+1):
            morph = seq[i:j]
            if morph in morph_lexicon:
                lattice[i].append((j, morph))
                has_known = True
        # UNK는 known이 없을 때만, 그리고 조건을 만족할 때만 추가
        if not has_known:
            for l in range(unk_max, unk_min - 1, -1):
                end = i + l
                if end <= N:
                    morph = seq[i:end]
                    if is_valid_unk(morph):
                        lattice[i].append((end, '<UNK>'))
                        break
    return lattice

def load_model(model_path):
    """
    학습된 HMM 모델 파라미터를 pickle 파일에서 로드합니다.
    Args:
        model_path (str): 모델 파일 경로
    Returns:
        tuple: (init_probs, trans_probs, emit_probs, tag_set)
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_morph_lexicon(emit_probs):
    """
    emit_probs에서 형태소 사전을 생성합니다.
    Args:
        emit_probs (dict): 방출 확률 테이블
    Returns:
        set: 형태소 집합
    """
    morph_lexicon = set()
    for tag in emit_probs:
        morph_lexicon.update(emit_probs[tag].keys())
    return morph_lexicon

def viterbi(seq, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, unk_min=2, unk_max=5, debug=False, length_penalty_alpha=0.8):
    """
    2차원 lattice 기반 Viterbi 알고리즘 (로그 확률 누적)
    Args:
        seq (str): 공백 없는 입력 문자열
        morph_lexicon (set): 형태소 사전
        tag_list (list): 품사 리스트
        init_probs, trans_probs, emit_probs: HMM 파라미터
        unk_min, unk_max: UNK 토큰 최소/최대 길이
        debug (bool): 디버깅 출력 여부
        length_penalty_alpha (float): 길이 패널티 알파값
    Returns:
        tuple: (최적 형태소 시퀀스(list), 품사 시퀀스(list), log_score(float))
    """
    N = len(seq)
    def safe_log(x):
        return math.log(x if x > 0 else 1e-12)
    def get_emit(tag, morph):
        if morph == '<UNK>':
            return emit_probs.get(tag, {}).get('<UNK>', UNK_EMIT_PROB)
        return emit_probs.get(tag, {}).get(morph, UNK_EMIT_PROB)
    lattice = build_lattice(seq, morph_lexicon, unk_min=unk_min, unk_max=unk_max)
    if debug:
        print("[lattice 후보]")
        for i in range(N):
            print(f"{i}: {[morph for _, morph in lattice[i]]}")
    V = [{} for _ in range(N+1)]
    V[0] = {}
    for tag in tag_list:
        V[0][tag] = (safe_log(init_probs.get(tag, UNK_EMIT_PROB)), (None, None, None))
    for i in range(N):
        for j, morph in lattice[i]:
            for tag in tag_list:
                emit = get_emit(tag, morph)
                log_emit = safe_log(emit)
                for prev_tag in V[i]:
                    prev_log_score, _ = V[i][prev_tag]
                    trans = trans_probs.get(prev_tag, {}).get(tag, UNK_EMIT_PROB)
                    log_trans = safe_log(trans)
                    log_score = prev_log_score + log_trans + log_emit
                    if tag not in V[j] or log_score > V[j][tag][0]:
                        V[j][tag] = (log_score, (i, prev_tag, morph))
    if not V[N]:
        return [], [], float('-inf')
    best_tag = max(V[N], key=lambda t: V[N][t][0])
    best_log_score = V[N][best_tag][0]
    tags, morphs = [], []
    idx, tag = N, best_tag
    while idx > 0:
        log_score, (prev_idx, prev_tag, morph) = V[idx][tag]
        tags.append(tag)
        morphs.append(morph)
        idx, tag = prev_idx, prev_tag
    morphs = morphs[::-1]
    tags = tags[::-1]
    norm_score = best_log_score / (len(morphs) ** length_penalty_alpha) if morphs else float('-inf')
    if debug:
        print("[Viterbi 최적 경로]")
        print("morphs:", morphs)
        print("tags:", tags)
        print("log_score (normalized):", norm_score)
    return morphs, tags, norm_score

def lattice_beam_search(seq, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, beam_width=5, unk_min=2, unk_max=5, debug=False, length_penalty_alpha=0.8):
    """
    Lattice + Beam Search 기반 형태소 분석 (로그 확률 누적)
    Args:
        seq (str): 공백 없는 입력 문자열
        morph_lexicon (set): 형태소 사전
        tag_list (list): 품사 리스트
        init_probs, trans_probs, emit_probs: HMM 파라미터
        beam_width (int): 빔 너비
        unk_min, unk_max: UNK 토큰 최소/최대 길이
        debug (bool): 디버깅 출력 여부
        length_penalty_alpha (float): 길이 패널티 알파값
    Returns:
        tuple: (최적 형태소 시퀀스(list), 품사 시퀀스(list), log_score(float))
    """
    N = len(seq)
    def safe_log(x):
        return math.log(x if x > 0 else 1e-12)
    def get_emit(tag, morph):
        if morph == '<UNK>':
            return emit_probs.get(tag, {}).get('<UNK>', UNK_EMIT_PROB)
        return emit_probs.get(tag, {}).get(morph, UNK_EMIT_PROB)
    lattice = build_lattice(seq, morph_lexicon, unk_min=unk_min, unk_max=unk_max)
    if debug:
        print("[lattice 후보]")
        for i in range(N):
            print(f"{i}: {[morph for _, morph in lattice[i]]}")
    beams = [[] for _ in range(N+1)]
    beams[0].append((0.0, [], []))
    for i in range(N):
        for log_score, path, tag_seq in beams[i]:
            for j, morph in lattice[i]:
                for tag in tag_list:
                    emit = get_emit(tag, morph)
                    log_emit = safe_log(emit)
                    if not tag_seq:
                        log_start = safe_log(init_probs.get(tag, UNK_EMIT_PROB))
                        log_trans = 0.0
                    else:
                        log_start = 0.0
                        log_trans = safe_log(trans_probs.get(tag_seq[-1], {}).get(tag, UNK_EMIT_PROB))
                    new_log_score = log_score + log_start + log_trans + log_emit
                    beams[j].append((new_log_score, path + [morph], tag_seq + [tag]))
        if i+1 <= N and beams[i+1]:
            beams[i+1] = sorted(beams[i+1], reverse=True, key=lambda x: x[0])[:beam_width]
    if not beams[N]:
        return [], [], float('-inf')
    # 길이 패널티 보정 적용
    best_candidate = max(
        beams[N],
        key=lambda x: x[0] / (len(x[1]) ** length_penalty_alpha) if len(x[1]) > 0 else float('-inf')
    )
    norm_score = best_candidate[0] / (len(best_candidate[1]) ** length_penalty_alpha) if best_candidate[1] else float('-inf')
    if debug:
        print("[Beam Search 최적 경로]")
        print("morphs:", best_candidate[1])
        print("tags:", best_candidate[2])
        print("log_score (normalized):", norm_score)
    return best_candidate[1], best_candidate[2], norm_score

def analyze(sentence, model, method=None, debug=False, unk_min=4, unk_max=5, length_penalty_alpha=0.4):
    """
    입력 문장을 다양한 방식으로 형태소 분석하여 품사 태깅 결과를 반환합니다.
    Args:
        sentence (str): 입력 문장
        model (tuple): (init_probs, trans_probs, emit_probs, tag_set)
        method (str, optional): 분석 방법 ('beam', 'viterbi')
        debug (bool): 디버깅 출력 여부
        unk_min, unk_max (int): UNK 토큰 최소/최대 길이
        length_penalty_alpha (float): 길이 패널티 알파값
    Returns:
        list: [(morph, tag), ...]
    """
    init_probs, trans_probs, emit_probs, tag_set = model
    cleaned = clean_text(sentence)
    tag_list = list(tag_set)
    morph_lexicon = get_morph_lexicon(emit_probs)
    seq = cleaned.replace(' ', '')

    if method == 'beam':
        result_morphs, result_tags, log_score = lattice_beam_search(
            seq, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, beam_width=5, unk_min=unk_min, unk_max=unk_max, debug=debug, length_penalty_alpha=length_penalty_alpha)
        if not result_morphs:
            print("lattice beam search 실패")
            return []
        return list(zip(result_morphs, result_tags))

    elif method == 'viterbi':
        morphs, tags, best_log_score = viterbi(seq, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, unk_min=unk_min, unk_max=unk_max, debug=debug, length_penalty_alpha=length_penalty_alpha)
        if best_log_score == float('-inf') or not morphs:
            print("lattice viterbi 실패")
            return []
        return list(zip(morphs, tags))

    else:
        print("[analyze] method는 'beam' 또는 'viterbi'만 지원합니다.")
        return []
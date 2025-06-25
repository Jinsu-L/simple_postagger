import pickle
import math
import re
from simple_postagger.preprocess import clean_text

# UNK 방출 확률 상수
UNK_EMIT_PROB = 1e-10

def is_valid_unk(morph):
    if len(morph) == 1:
        if re.fullmatch(r'[은는이가을를에의]', morph):
            return False
        if re.fullmatch(r'[0-9]|[^\w가-힣]', morph):
            return False
    return True

def build_lattice(seq, morph_lexicon, unk_min=4, unk_max=5):
    original_seq = seq
    whitespace_indices = [m.start() for m in re.finditer(' ', original_seq)]
    cleaned_seq = seq.replace(' ', '')
    N = len(cleaned_seq)
    lattice = [[] for _ in range(N+1)]
    for i in range(N):
        has_known = False
        for j in range(i+1, N+1):
            orig_start, orig_end, pos = 0, 0, 0
            for idx in range(len(original_seq)):
                if original_seq[idx] == ' ':
                    continue
                if pos == i:
                    orig_start = idx
                if pos == j - 1:
                    orig_end = idx + 1
                    break
                pos += 1
            if any(ws in range(orig_start, orig_end) for ws in whitespace_indices):
                continue
            morph = cleaned_seq[i:j]
            if morph in morph_lexicon:
                lattice[i].append((j, morph))
                has_known = True
        if not has_known:
            for l in range(unk_max, unk_min - 1, -1):
                end = i + l
                if end <= N:
                    morph = cleaned_seq[i:end]
                    if is_valid_unk(morph):
                        lattice[i].append((end, '<UNK>'))
                        break
    return lattice

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_morph_lexicon(emit_probs):
    morph_lexicon = set()
    for tag in emit_probs:
        morph_lexicon.update(emit_probs[tag].keys())
    return morph_lexicon

def viterbi(seq, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, unk_min=2, unk_max=5, debug=False, length_penalty_alpha=0.8):
    cleaned_seq = seq.replace(' ', '')
    N = len(cleaned_seq)
    def safe_log(x):
        return math.log(x if x > 0 else 1e-12)
    def get_emit(tag, morph):
        if morph == '<UNK>':
            return UNK_EMIT_PROB * 0.1  # 더 불리하게
        return emit_probs.get(tag, {}).get(morph, UNK_EMIT_PROB)

    lattice = build_lattice(seq, morph_lexicon, unk_min=unk_min, unk_max=unk_max)
    if debug:
        print("[lattice 후보]")
        for i in range(N):
            print(f"{i}: {[morph for _, morph in lattice[i]]}")
    V = [{} for _ in range(N+1)]
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
    cleaned_seq = seq.replace(' ', '')
    N = len(cleaned_seq)
    def safe_log(x):
        return math.log(x if x > 0 else 1e-12)
    def get_emit(tag, morph):
        if morph == '<UNK>':
            return UNK_EMIT_PROB * 0.1  # 더 불리하게
        return emit_probs.get(tag, {}).get(morph, UNK_EMIT_PROB)
    lattice = build_lattice(seq, morph_lexicon, unk_min=unk_min, unk_max=unk_max)
    if debug:
        print("[lattice 후보]")
        for i in range(len(lattice)):
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
    best_seq = max(
        beams[N],
        key=lambda x: x[0] / (len(x[1]) ** length_penalty_alpha) if len(x[1]) > 0 else float('-inf'),
        default=([], [], [])
    )
    if not best_seq or not best_seq[1]:
        return [], [], float('-inf')
    norm_score = best_seq[0] / (len(best_seq[1]) ** length_penalty_alpha)
    if debug:
        print("[Beam Search 최적 경로]")
        print("morphs:", best_seq[1])
        print("tags:", best_seq[2])
        print("log_score (normalized):", norm_score)
    return best_seq[1], best_seq[2], norm_score

def analyze(sentence, model, method='viterbi', debug=False, unk_min=4, unk_max=5, length_penalty_alpha=0.4):
    init_probs, trans_probs, emit_probs, tag_set = model
    cleaned = clean_text(sentence)
    tag_list = list(tag_set)
    morph_lexicon = get_morph_lexicon(emit_probs)
    if method == 'viterbi':
        morphs, tags, best_log_score = viterbi(cleaned, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, unk_min=unk_min, unk_max=unk_max, debug=debug, length_penalty_alpha=length_penalty_alpha)
        if best_log_score == float('-inf') or not morphs:
            print("lattice viterbi 실패")
            return []
        return list(zip(morphs, tags))
    elif method == 'beam':
        morphs, tags, best_log_score = lattice_beam_search(cleaned, morph_lexicon, tag_list, init_probs, trans_probs, emit_probs, beam_width=5, unk_min=unk_min, unk_max=unk_max, debug=debug, length_penalty_alpha=length_penalty_alpha)
        if best_log_score == float('-inf') or not morphs:
            print("lattice beam search 실패")
            return []
        return list(zip(morphs, tags))
    else:
        print("[analyze] method는 'beam' 또는 'viterbi'만 지원합니다.")
        return []
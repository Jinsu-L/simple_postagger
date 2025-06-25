"""
viterbi.py
- HMM 기반 비터비 알고리즘 구현
"""

def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    비터비 알고리즘으로 최적의 품사 시퀀스 추정

    Args:
        obs (list): 관측 시퀀스(형태소 리스트)
        states (list): 품사 리스트
        start_p (dict): 초기 확률
        trans_p (dict): 전이 확률
        emit_p (dict): 방출 확률

    Returns:
        tuple: (최적 품사 시퀀스(list), score(float))
    """
    V = [{}]
    path = {}

    for y in states:
        V[0][y] = start_p.get(y, 1e-8) * emit_p.get(y, {}).get(obs[0], 1e-8)
        path[y] = [y]
        
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}
        for y in states:
            (prob, state) = max(
                (V[t-1][y0] * trans_p.get(y0, {}).get(y, 1e-8) * emit_p.get(y, {}).get(obs[t], 1e-8), y0)
                for y0 in states
            )
            V[t][y] = prob
            new_path[y] = path[state] + [y]
        path = new_path
        
    n = len(obs) - 1
    (prob, state) = max((V[n][y], y) for y in states)
    return path[state], prob

if __name__ == '__main__':
    obs = ['나는', '학생이다']
    states = ['NOUN', 'VERB']
    start_p = {'NOUN': 0.6, 'VERB': 0.4}
    trans_p = {'NOUN': {'NOUN': 0.1, 'VERB': 0.9}, 'VERB': {'NOUN': 0.8, 'VERB': 0.2}}
    emit_p = {'NOUN': {'나는': 0.5, '학생이다': 0.5}, 'VERB': {'나는': 0.1, '학생이다': 0.9}}
    tags, score = viterbi(obs, states, start_p, trans_p, emit_p)
    print('예측 품사 시퀀스:', tags)
    print('Score:', score) 
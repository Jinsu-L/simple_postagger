"""
dictionary.py
- 형태소-품사 사전 구축 및 관련 유틸리티
"""
from collections import Counter

def build_dictionary(sentences):
    """
    형태소-품사 쌍 빈도 집계

    Args:
        sentences (list): List of (sentence, [(morph, tag), ...]) tuples.

    Returns:
        Counter: (morph, tag) 쌍의 빈도 Counter
    """
    morph_tag_counter = Counter()
    for _, morphs in sentences:
        for morph, tag in morphs:
            morph_tag_counter[(morph, tag)] += 1
    return morph_tag_counter 
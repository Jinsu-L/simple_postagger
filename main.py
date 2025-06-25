import argparse
import pickle
from simple_postagger.preprocess import load_json_corpus, clean_text, save_preprocessed_data
from simple_postagger.train import estimate_hmm_parameters, save_model
from simple_postagger.analyzer import load_model, analyze

def preprocess(json_path, output_path):
    data = load_json_corpus(json_path)
    data = [(clean_text(sent), [(clean_text(m), t) for m, t in morphs]) for sent, morphs in data]
    save_preprocessed_data(data, output_path)
    print(f'전처리 완료: {len(data)} 문장')

def train(preprocessed_path, model_path):
    with open(preprocessed_path, 'rb') as f:
        sentences = pickle.load(f)
    init_p, trans_p, emit_p, tag_set = estimate_hmm_parameters(sentences)
    save_model((init_p, trans_p, emit_p, tag_set), model_path)
    print('모델 학습 및 저장 완료')

def analyze_sentence(model_path, sentence):
    model = load_model(model_path)
    result = analyze(sentence, model)
    print('분석 결과:')
    for morph, tag in result:
        print(f'{morph}/{tag}', end=' ')
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='세종 코퍼스 기반 HMM 형태소 분석기')
    parser.add_argument('--preprocess', nargs=2, metavar=('json_path', 'output_path'), help='JSON → 전처리 데이터(pkl)')
    parser.add_argument('--train', nargs=2, metavar=('preprocessed_path', 'model_path'), help='전처리 데이터 → HMM 모델(pkl)')
    parser.add_argument('--analyze', nargs=2, metavar=('model_path', 'sentence'), help='임의 문장 분석')
    parser.add_argument('--evaluate', nargs=2, metavar=('model_path', 'preprocessed_path'), help='모델 평가')
    args = parser.parse_args()

    if args.preprocess:
        preprocess(*args.preprocess)
    if args.train:
        train(*args.train)
    if args.analyze:
        analyze_sentence(*args.analyze)
# Simple POSTagger

간단한 HMM 기반 한국어 형태소 분석기 (세종 말뭉치 기반)

## 패키지 설치

```
pip install -r requirements.txt
```

## 데이터 준비
- 학습용 세종 코퍼스 JSON 파일을 `data/` 폴더에 넣으세요.

## 실행 예시

1. **전처리**
    ```
    python main.py --preprocess data/NIKL_MP(v1.1)/SXMP1902008031.json data/NIKL_MP(v1.1)/preprocessed.pkl
    ```
2. **학습**
    ```
    python main.py --train data/NIKL_MP(v1.1)/preprocessed.pkl data/NIKL_MP(v1.1)/hmm_model.pkl
    ```
3. **분석**
    ```
    python main.py --analyze data/NIKL_MP(v1.1)/hmm_model.pkl "아버지 가방에 들어가신다"
    ```

## 사용 예시
- 사용 예시는 `example.ipynb`를 참고.
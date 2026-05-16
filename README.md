# Cine21 영화 데이터 PCA·회귀 분석 프로젝트

Cine21 영화 데이터를 수집·전처리하고 PCA, 상관분석, ARIMA, 선형회귀를 활용해 영화 평점과 흥행 지표의 관계를 분석한 데이터 분석 프로젝트입니다.

![프로젝트 포스터](docs/poster_202212004_hwang_minseo.png)

## 프로젝트 개요

- **주제:** PCA 기반 차원 축소와 회귀 분석을 통한 영화 평점·흥행 지표 관계 분석
- **데이터 출처:** Cine21 영화 정보 페이지 (저장된 CSV 사용)
- **분석 대상:** 영화 메타데이터(개봉일, 등급, 상영 시간, 누적 관객 수 등) 및 평점
- **데이터 규모:** 약 150행, 다수의 메타컬럼(원본 CSV 참조)
- **주요 목표:** 수치형 변수의 차원 축소(PCA), 변수 간 상관성 비교, 시계열 예측(ARIMA), PC1 기반 회귀모형 평가

## 주요 기능

- Cine21 데이터 기반 기본 정보 수집(크롤러 포함)
- 결측값 처리, 이상치 처리, 중복 제거
- MinMaxScaler 기반 정규화 및 PCA 수행
- 피어슨/스피어만/켄달 상관계수 비교
- ARIMA 기반 월별 평균 평점 예측(가능한 경우)
- 제1주성분(PC1) 기반 선형회귀 분석 및 평점 예측 비교

## 사용 기술

- Python
- Jupyter Notebook / Google Colab
- pandas, numpy
- requests, BeautifulSoup4
- scikit-learn, statsmodels
- matplotlib

## 프로젝트 구조

```text
movie-pca-rating-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── hms202212004.csv
├── notebooks/
│   └── hms_project_202212004.ipynb
├── src/
│   └── hms_project_202212004.py
├── outputs/
│   └── .gitkeep
└── docs/
    ├── poster_202212004_hwang_minseo.png
    ├── presentation_202212004_hwang_minseo.pptx
    └── report_202212004_hwang_minseo.hwp
```

## 데이터 컬럼

아래는 주요 컬럼(원본 CSV에 자세한 정보 포함)입니다.

| 컬럼명 | 설명 |
|---|---|
| movie_id | 영화 고유 ID |
| title | 영화 제목 |
| rank | 수집 순위 |
| title_ko | 국문 제목 |
| title_en | 영문 제목 |
| year | 제작 연도 |
| open_date_detail | 개봉일 |
| rating | 관람 등급 |
| runtime_min | 상영 시간 |
| audience_total | 누적 관객 수 |
| genres | 장르 |
| country | 제작 국가 |
| directors | 감독 |
| num_directors | 감독 수 |
| cast_main | 주요 출연진 |
| num_cast_main | 주요 출연진 수 |
| cine21_score_detail | Cine21 전문가 평점 |
| netizen_score | 관객 평점 |

## 실행 방법

1. 패키지 설치

```bash
pip install -r requirements.txt
```

2. 노트북 실행

```bash
jupyter notebook notebooks/hms_project_202212004.ipynb
```

3. Python 스크립트 실행

```bash
python src/hms_project_202212004.py
```

4. 실행 결과

실행하면 `outputs/` 폴더에 다음 CSV 파일들이 생성됩니다 (스크립트 실행 환경에 따라 일부 단계는 건너뛸 수 있음):

- `hms_movies_clean.csv` — 결측·이상치 처리 후 정리된 데이터
- `hms_movies_scaled.csv` — 정규화(Scaled)된 수치형 변수 포함 데이터
- `hms_movies_pca.csv` — PCA 변환 결과(주성분) 포함 데이터

로컬 환경에서 그래프 한글이 깨질 경우 시스템에 `NanumGothic` 등 한글 폰트를 설치하면 시각화에서 한글이 정상 표시됩니다. 스크립트는 폰트가 없어도 중단되지 않습니다.

## 분석 흐름

데이터 수집(선택) → 데이터 병합 → 결측값 처리 → 이상치 처리 → 중복 제거 → 정규화 → PCA → 상관관계 분석 → 시계열 예측(ARIMA, 가능 시) → 선형회귀 분석

## 결과 요약

- PCA로 수치형 변수(`runtime_min`, `audience_total`, `cine21_score_detail`, `netizen_score`)의 축약 가능성을 확인했습니다.
- 피어슨/스피어만/켄달 상관계수로 변수 간 관계를 비교했습니다.
- ARIMA로 월별 평균 평점의 단기 추세를 예측(데이터가 부족하면 해당 단계는 건너뜁니다).
- 제1주성분(PC1)을 설명변수로 사용한 선형회귀로 실제 평점과 예측 평점의 관계를 비교했습니다.

## 문서 자료

`docs/` 폴더에 프로젝트 포스터, 발표자료, 보고서가 포함되어 있습니다. (원본 파일은 변경하지 마십시오.)

## 추가 메모

- 이 저장소의 분석 주제 및 원본 데이터(`data/hms202212004.csv`)는 보존되어야 합니다.

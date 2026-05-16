# GitHub 업로드 가이드

## 1. 추천 저장소 이름

```text
movie-pca-rating-analysis
```

## 2. 저장소 설명

```text
Cine21 영화 데이터를 수집·전처리하고 PCA, 상관분석, ARIMA, 선형회귀로 영화 흥행과 평점을 분석한 데이터 분석 프로젝트
```

## 3. GitHub에서 새 저장소 만들기

1. GitHub 접속
2. New repository 클릭
3. Repository name에 `movie-pca-rating-analysis` 입력
4. Description에 위 저장소 설명 입력
5. Public 선택
6. README, .gitignore, license는 체크하지 않기
7. Create repository 클릭

## 4. 터미널 업로드 명령어

압축을 푼 폴더 안에서 아래 명령어를 실행합니다.

```bash
git init
git add .
git commit -m "Add movie PCA rating analysis project"
git branch -M main
git remote add origin https://github.com/allen8524/movie-pca-rating-analysis.git
git push -u origin main
```

저장소 이름을 다르게 만들었다면 마지막 주소만 바꾸면 됩니다.

## 5. 포트폴리오 카드용 문구

제목:

```text
영화 데이터 PCA·회귀 분석 프로젝트
```

한 줄 설명:

```text
Cine21 영화 데이터를 수집·전처리하고 PCA, 상관분석, ARIMA, 선형회귀를 적용해 영화 평점과 흥행 지표를 분석한 데이터 분석 프로젝트입니다.
```

사용 기술:

```text
Python, pandas, BeautifulSoup, scikit-learn, statsmodels, matplotlib, Google Colab
```

핵심 기능:

```text
영화 데이터 크롤링, 결측값·이상치 처리, 정규화, PCA, 상관관계 분석, ARIMA 평점 예측, PC1 기반 선형회귀 분석
```

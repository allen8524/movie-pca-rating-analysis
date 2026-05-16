df_na["rating"] = df_na["rating"].fillna("정보없음")
df_na["runtime_min"] = df_na["runtime_min"].fillna(df_na["runtime_min"].median())
df_na["cine21_score_detail"] = df_na["cine21_score_detail"].fillna(df_na["cine21_score_detail"].mean())
"""Movie PCA & rating analysis — 포트폴리오용 실행 스크립트

이 스크립트는 저장된 CSV(`data/hms202212004.csv`)를 기준으로
데이터 전처리, 정규화, PCA, 상관분석, ARIMA(선택), 선형회귀를 실행하고
결과물을 `outputs/`에 CSV로 저장합니다.

주의:
- 원본 데이터 `data/hms202212004.csv`를 삭제하지 마십시오.
"""

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "hms202212004.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def preprocess_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 안전하게 컬럼 제거
    df = df.drop(columns=["title_en"], errors="ignore")

    # 범주형 기본값
    for col in ["rating", "country", "directors", "cast_main"]:
        if col in df.columns:
            df[col] = df[col].fillna("정보없음")

    # 수치형 결측값
    if "runtime_min" in df.columns:
        df["runtime_min"] = df["runtime_min"].fillna(df["runtime_min"].median())
    if "audience_total" in df.columns:
        df["audience_total"] = df["audience_total"].fillna(df["audience_total"].median())
    if "cine21_score_detail" in df.columns:
        df["cine21_score_detail"] = df["cine21_score_detail"].fillna(df["cine21_score_detail"].mean())
    if "netizen_score" in df.columns:
        df["netizen_score"] = df["netizen_score"].fillna(df["netizen_score"].mean())

    return df


def clip_outliers_series(s: pd.Series) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return s.clip(lower=low, upper=high)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"]:
        if col in df.columns:
            df[col] = clip_outliers_series(df[col])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "movie_id" in df.columns:
        df = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = [c for c in ["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"] if c in df.columns]
    if not num_cols:
        print("정규화할 수치형 컬럼이 없습니다. 건너뜁니다.")
        return df
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def run_pca(df_scaled: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in ["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"] if c in df_scaled.columns]
    if not feature_cols:
        print("PCA 실행 조건인 피처가 없습니다. 건너뜁니다.")
        return pd.DataFrame()
    X = df_scaled[feature_cols].values
    pca = PCA()
    X_pca = pca.fit_transform(X)
    pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)
    if "movie_id" in df_scaled.columns:
        df_pca["movie_id"] = df_scaled["movie_id"].values
    if "title" in df_scaled.columns:
        df_pca["title"] = df_scaled["title"].values
    return df_pca


def analyze_correlation(df_scaled: pd.DataFrame) -> None:
    cols = [c for c in ["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"] if c in df_scaled.columns]
    if len(cols) < 2:
        print("상관분석에 필요한 컬럼이 부족합니다. 건너뜁니다.")
        return
    data = df_scaled[cols]
    print("=== Pearson ===")
    print(data.corr(method="pearson"))
    print("\n=== Spearman ===")
    print(data.corr(method="spearman"))
    print("\n=== Kendall ===")
    print(data.corr(method="kendall"))


def run_arima_forecast(df_clean: pd.DataFrame) -> None:
    if ARIMA is None:
        print("statsmodels ARIMA를 불러올 수 없습니다. ARIMA 단계 생략합니다.")
        return
    if "open_date_detail" not in df_clean.columns:
        print("개봉일 컬럼이 없어 ARIMA를 실행할 수 없습니다.")
        return

    def extract_date(text):
        if pd.isna(text):
            return pd.NaT
        s = str(text)
        m = re.search(r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}", s)
        if not m:
            return pd.NaT
        d = m.group(0).replace('.', '-').replace('/', '-')
        return pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")

    df = df_clean.copy()
    df["open_date"] = df["open_date_detail"].apply(extract_date)
    df = df.dropna(subset=["open_date"]) if "open_date" in df.columns else df
    rating_col = "cine21_score_detail"
    if rating_col not in df.columns:
        print("ARIMA에 필요한 평점 컬럼이 없습니다. 건너뜁니다.")
        return
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])
    if df.empty or "open_date" not in df.columns:
        print("ARIMA 실행을 위한 적절한 시계열 데이터를 만들 수 없습니다. 건너뜁니다.")
        return

    monthly = (
        df.set_index("open_date").groupby(pd.Grouper(freq="ME"))[rating_col].mean().sort_index()
    )
    if monthly.dropna().shape[0] < 6:
        print("월별 데이터가 충분하지 않아 ARIMA를 건너뜁니다.")
        return

    monthly = monthly.asfreq("ME").interpolate()
    try:
        model = ARIMA(monthly, order=(1, 1, 1))
        fit = model.fit()
        forecast_res = fit.get_forecast(steps=1)
        pred_mean = forecast_res.predicted_mean
        print("최근 5개월 평균 평점")
        print(monthly.tail())
        print("다음달 예상 평균 평점:", float(pred_mean.iloc[0]))
    except Exception as exc:
        print("ARIMA 실행 중 오류 발생, 건너뜁니다:", exc)


def run_linear_regression(df_scaled: pd.DataFrame, df_pca: pd.DataFrame) -> None:
    if df_pca.empty or "PC1" not in df_pca.columns:
        print("PC1 데이터가 없어 회귀를 실행할 수 없습니다. 건너뜁니다.")
        return
    if "cine21_score_detail" not in df_scaled.columns:
        print("평점 컬럼이 없어 회귀를 실행할 수 없습니다. 건너뜁니다.")
        return

    X = df_pca[["PC1"]].values
    y = df_scaled.loc[df_pca.index, "cine21_score_detail"].values if "movie_id" not in df_pca.columns else df_scaled.set_index("movie_id").loc[df_pca["movie_id"], "cine21_score_detail"].values

    model = LinearRegression()
    try:
        model.fit(X, y)
        y_pred = model.predict(X)
        print("회귀계수(기울기):", float(model.coef_[0]))
        print("절편:", float(model.intercept_))
        print("결정계수 R^2:", r2_score(y, y_pred))
        print("평균제곱오차 MSE:", mean_squared_error(y, y_pred))
    except Exception as exc:
        print("회귀분석 중 오류 발생, 건너뜁니다:", exc)


def plot_actual_vs_predicted(y, y_pred, out_path: Path):
    try:
        plt.figure(figsize=(5, 5))
        plt.scatter(y, y_pred, alpha=0.6)
        min_v = min(np.nanmin(y), np.nanmin(y_pred))
        max_v = max(np.nanmax(y), np.nanmax(y_pred))
        plt.plot([min_v, max_v], [min_v, max_v], color="red")
        plt.xlabel("실제 평점")
        plt.ylabel("예측 평점")
        plt.title("실제 평점 vs 예측 평점")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 로드
    try:
        df_raw = load_data()
        print("원본 데이터 로드 완료, 행:", len(df_raw))
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    # 전처리
    df_clean = preprocess_missing_values(df_raw)
    df_clean = remove_outliers(df_clean)
    df_clean = remove_duplicates(df_clean)
    df_clean.to_csv(OUTPUT_DIR / "hms_movies_clean.csv", index=False, encoding="utf-8-sig")
    print("정리된 데이터 저장:", OUTPUT_DIR / "hms_movies_clean.csv")

    # 정규화
    df_scaled = normalize_data(df_clean)
    df_scaled.to_csv(OUTPUT_DIR / "hms_movies_scaled.csv", index=False, encoding="utf-8-sig")
    print("정규화된 데이터 저장:", OUTPUT_DIR / "hms_movies_scaled.csv")

    # PCA
    df_pca = run_pca(df_scaled)
    if not df_pca.empty:
        df_pca.to_csv(OUTPUT_DIR / "hms_movies_pca.csv", index=False, encoding="utf-8-sig")
        print("PCA 결과 저장:", OUTPUT_DIR / "hms_movies_pca.csv")
    else:
        print("PCA 결과가 없어 저장하지 않았습니다.")

    # 상관분석
    analyze_correlation(df_scaled)

    # ARIMA 예측 (가능한 경우)
    try:
        run_arima_forecast(df_clean)
    except Exception as exc:
        print("ARIMA 단계 중 예외:", exc)

    # 회귀분석
    try:
        run_linear_regression(df_scaled, df_pca)
    except Exception as exc:
        print("회귀 단계 중 예외:", exc)


if __name__ == "__main__":
    main()

df_scaled.to_csv("hms_movies_scaled.csv", index=False, encoding="utf-8-sig")
df_scaled.head()

"""# [HMS Project Code-8] 주성분/설명력 구하기"""

import pandas as pd
from sklearn.decomposition import PCA

df_scaled = pd.read_csv("hms_movies_scaled.csv")

feature_cols = ["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"]
X = df_scaled[feature_cols]

pca = PCA()
X_pca = pca.fit_transform(X)

explained = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

print("=== PCA 설명력 ===")
for i, r in enumerate(explained, start=1):
    print(f"제{i}주성분의 설명력 : {r*100:.2f}%")

print("\n=== PCA 고유값 ===")
for i, v in enumerate(eigenvalues, start=1):
    print(f"제{i}주성분의 고유값 : {v:.6f}")

pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pc_cols)
df_pca["movie_id"] = df_scaled["movie_id"]
df_pca["title"] = df_scaled["title"]

df_pca.to_csv("hms_movies_pca.csv", index=False, encoding="utf-8-sig")
df_pca.head()

"""# [HMS Project Code-9] 상관관계 분석 : 피어슨, 스피어만, 켄달"""

import pandas as pd

df_scaled = pd.read_csv("hms_movies_scaled.csv")

gsm_totalData_normalized = df_scaled[["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"]]

coef_p = gsm_totalData_normalized.corr(method="pearson")
print("=== 피어슨 상관계수 ===")
print(coef_p)

print()

coef_s = gsm_totalData_normalized.corr(method="spearman")
print("=== 스피어만 상관계수 ===")
print(coef_s)

print()

coef_k = gsm_totalData_normalized.corr(method="kendall")
print("=== 켄달 상관계수 ===")
print(coef_k)

"""# [HMS Project Code-10-1] 그래프 구현을 위한 한글 폰트 설치"""

# Colab only: apt-get -y install fonts-nanum
# Colab only: fc-cache -fv
# Colab only: rm -rf /root/.cache/matplotlib/

"""# [HMS Project Code-10-2] 월별 개봉 영화를 시계열 분석하여 다음달 영화의 평점을 예측하는 코드"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from matplotlib import rcParams
rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False


df = pd.read_csv("hms_movies_clean.csv")

def extract_date(text):
    if pd.isna(text):
        return pd.NaT
    s = str(text)
    m = re.search(r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}", s)
    if not m:
        return pd.NaT
    d = m.group(0).replace(".", "-")
    return pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")

df["open_date"] = df["open_date_detail"].apply(extract_date)
df = df.dropna(subset=["open_date"])

rating_col = "cine21_score_detail"
df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
df = df.dropna(subset=[rating_col])

monthly = (
    df.set_index("open_date")
      .groupby(pd.Grouper(freq="ME"))[rating_col]
      .mean()
      .sort_index()
)

monthly = monthly.asfreq("ME")
monthly = monthly.interpolate()

model = ARIMA(monthly, order=(1, 1, 1))
fit = model.fit()

forecast_res = fit.get_forecast(steps=1)
pred_mean = forecast_res.predicted_mean

print("최근 5개월 평균 평점")
print(monthly.tail())
print()
print("다음달 예상 평균 평점:", float(pred_mean.iloc[0]))

plt.figure(figsize=(10, 4))
plt.plot(monthly.index, monthly, label="실제 월별 평균 평점")
plt.plot(pred_mean.index, pred_mean, marker="o", label="예측(다음달)")
plt.xlabel("월")
plt.ylabel("평점")
plt.title("월별 평균 평점과 다음달 예측값")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""# [HMS Project Code-11] 주성분(PC1)을 이용한 선형회귀분석"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df_scaled = pd.read_csv("hms_movies_scaled.csv")
df_pca = pd.read_csv("hms_movies_pca.csv")

X = df_pca[["PC1"]]
y = df_scaled["cine21_score_detail"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("회귀계수(기울기):", float(model.coef_[0]))
print("절편:", float(model.intercept_))
print("결정계수 R^2:", r2_score(y, y_pred))
print("평균제곱오차 MSE:", mean_squared_error(y, y_pred))

plt.figure(figsize=(6, 4))
plt.scatter(X["PC1"], y, alpha=0.5, label="실제 평점")
plt.plot(X["PC1"], y_pred, label="회귀 직선")
plt.xlabel("제1주성분 PC1")
plt.ylabel("cine21 평점")
plt.title("제1주성분을 이용한 평점 선형회귀분석")
plt.legend()
plt.tight_layout()
plt.show()

"""# [HMS Project Code-12] 제1주성분(PC1)을 이용한 영화 평점 예측(선형회귀분석)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

plt.figure(figsize=(5, 5))
plt.scatter(y, y_pred, alpha=0.5)
min_v = min(y.min(), y_pred.min())
max_v = max(y.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v])
plt.xlabel("실제 평점(정규화)")
plt.ylabel("예측 평점(정규화)")
plt.title("실제 평점 vs 예측 평점")
plt.tight_layout()
plt.show()

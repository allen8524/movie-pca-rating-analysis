"""Offline execution script for the Cine21 movie PCA rating analysis project.

The script loads the saved CSV file from data/hms202212004.csv, cleans and
normalizes the data, runs PCA, prints correlation results, and optionally runs
ARIMA and linear regression when enough local data is available.
"""

from pathlib import Path
import re
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "hms202212004.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "docs" / "images"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.txt"

NUMERIC_COLUMNS = ["runtime_min", "audience_total", "cine21_score_detail", "netizen_score"]

warnings.filterwarnings("ignore")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def preprocess_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["title_en"], errors="ignore")

    for column in ["rating", "country", "directors", "cast_main"]:
        if column in df.columns:
            df[column] = df[column].fillna("정보없음")

    numeric_fill_rules = {
        "runtime_min": "median",
        "audience_total": "median",
        "cine21_score_detail": "mean",
        "netizen_score": "mean",
    }

    for column, method in numeric_fill_rules.items():
        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(df[column], errors="coerce")
        fill_value = df[column].median() if method == "median" else df[column].mean()
        if pd.isna(fill_value):
            print(f"{column} 컬럼에 유효한 값이 없어 0으로 채웁니다.")
            fill_value = 0
        df[column] = df[column].fillna(fill_value)

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        valid_series = series.dropna()
        if valid_series.empty:
            print(f"{column} 컬럼에 유효한 값이 없어 이상치 처리를 건너뜁니다.")
            continue

        q1 = valid_series.quantile(0.25)
        q3 = valid_series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[column] = series.clip(lower=lower_bound, upper=upper_bound)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "movie_id" in df.columns:
        return df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [column for column in NUMERIC_COLUMNS if column in df.columns]

    if not numeric_columns:
        print("정규화할 수치형 컬럼이 없습니다. 정규화 단계를 건너뜁니다.")
        return df

    numeric_data = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_data.dropna().empty:
        print("정규화할 유효한 수치형 데이터가 없습니다. 정규화 단계를 건너뜁니다.")
        return df

    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(numeric_data)
    return df


def run_pca(df_scaled: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feature_columns = [column for column in NUMERIC_COLUMNS if column in df_scaled.columns]

    if not feature_columns:
        print("PCA에 필요한 수치형 컬럼이 없습니다. PCA 단계를 건너뜁니다.")
        return pd.DataFrame(), np.array([])

    pca_data = df_scaled[feature_columns].apply(pd.to_numeric, errors="coerce")
    pca_data = pca_data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pca_data) < 2:
        print("PCA에 필요한 데이터가 부족합니다. PCA 단계를 건너뜁니다.")
        return pd.DataFrame(), np.array([])

    n_components = min(len(feature_columns), len(pca_data))
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(pca_data)
    component_columns = [f"PC{i + 1}" for i in range(transformed.shape[1])]
    df_pca = pd.DataFrame(transformed, columns=component_columns, index=pca_data.index)

    if "movie_id" in df_scaled.columns:
        df_pca["movie_id"] = df_scaled.loc[pca_data.index, "movie_id"].values
    if "title" in df_scaled.columns:
        df_pca["title"] = df_scaled.loc[pca_data.index, "title"].values

    explained_variance_ratio = pca.explained_variance_ratio_
    print("PCA Explained Variance Ratio:")
    for component, ratio in zip(component_columns, explained_variance_ratio):
        print(f"{component}: {ratio:.6f}")

    return df_pca, explained_variance_ratio


def analyze_correlation(df_scaled: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in NUMERIC_COLUMNS if column in df_scaled.columns]

    if len(columns) < 2:
        print("상관분석에 필요한 컬럼이 부족합니다. 상관분석을 건너뜁니다.")
        return pd.DataFrame()

    correlation_data = df_scaled[columns].apply(pd.to_numeric, errors="coerce")
    correlation_data = correlation_data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(correlation_data) < 2:
        print("상관분석에 필요한 데이터가 부족합니다. 상관분석을 건너뜁니다.")
        return pd.DataFrame()

    pearson_corr = correlation_data.corr(method="pearson")
    print("=== Pearson ===")
    print(pearson_corr)
    print("\n=== Spearman ===")
    print(correlation_data.corr(method="spearman"))
    print("\n=== Kendall ===")
    print(correlation_data.corr(method="kendall"))

    return pearson_corr


def run_arima_forecast(df_clean: pd.DataFrame) -> dict:
    def skip(reason: str) -> dict:
        print(reason)
        return {"forecast": None, "skipped": reason}

    if ARIMA is None:
        return skip("statsmodels ARIMA를 불러올 수 없습니다. ARIMA 단계를 건너뜁니다.")

    if "open_date_detail" not in df_clean.columns:
        return skip("개봉일 컬럼이 없어 ARIMA 단계를 건너뜁니다.")

    rating_column = "cine21_score_detail"
    if rating_column not in df_clean.columns:
        return skip("평점 컬럼이 없어 ARIMA 단계를 건너뜁니다.")

    date_pattern = re.compile(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})")
    df = df_clean.copy()
    date_text = df["open_date_detail"].astype(str).str.extract(date_pattern, expand=False)
    date_text = date_text.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    df["open_date"] = pd.to_datetime(date_text, errors="coerce")
    df[rating_column] = pd.to_numeric(df[rating_column], errors="coerce")
    df = df.dropna(subset=["open_date", rating_column])

    if len(df) < 6:
        return skip("ARIMA에 필요한 관측치가 부족합니다. ARIMA 단계를 건너뜁니다.")

    month_end_freq = "ME"
    try:
        pd.date_range("2000-01-01", periods=1, freq=month_end_freq)
    except ValueError:
        month_end_freq = "M"

    monthly_rating = (
        df.set_index("open_date")
        .sort_index()
        .resample(month_end_freq)[rating_column]
        .mean()
        .dropna()
    )

    if len(monthly_rating) < 6:
        return skip("월별 평점 데이터가 부족합니다. ARIMA 단계를 건너뜁니다.")

    monthly_rating = monthly_rating.asfreq(month_end_freq).interpolate()

    try:
        model = ARIMA(monthly_rating, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = float(fitted_model.get_forecast(steps=1).predicted_mean.iloc[0])
    except Exception as exc:
        return skip(f"ARIMA 실행 중 오류가 발생했습니다. ARIMA 단계를 건너뜁니다: {exc}")

    print("최근 5개월 평균 평점")
    print(monthly_rating.tail())
    print(f"ARIMA Forecast: {forecast:.6f}")
    return {"forecast": forecast, "skipped": None}


def run_linear_regression(df_scaled: pd.DataFrame, df_pca: pd.DataFrame) -> dict:
    def skip(reason: str) -> dict:
        print(reason)
        return {"r2": None, "mse": None, "y_actual": None, "y_pred": None, "skipped": reason}

    if df_pca.empty or "PC1" not in df_pca.columns:
        return skip("PC1 데이터가 없어 회귀분석을 건너뜁니다.")

    target_column = "cine21_score_detail"
    if target_column not in df_scaled.columns:
        return skip("평점 컬럼이 없어 회귀분석을 건너뜁니다.")

    common_index = df_pca.index.intersection(df_scaled.index)
    if len(common_index) < 2:
        return skip("회귀분석에 필요한 데이터가 부족합니다. 회귀분석을 건너뜁니다.")

    regression_data = pd.DataFrame(
        {
            "PC1": pd.to_numeric(df_pca.loc[common_index, "PC1"], errors="coerce"),
            target_column: pd.to_numeric(df_scaled.loc[common_index, target_column], errors="coerce"),
        }
    )
    regression_data = regression_data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(regression_data) < 2 or regression_data["PC1"].nunique() < 2:
        return skip("회귀분석에 필요한 유효한 데이터가 부족합니다. 회귀분석을 건너뜁니다.")

    x = regression_data[["PC1"]].values
    y_actual = regression_data[target_column].values

    model = LinearRegression()
    model.fit(x, y_actual)
    y_pred = model.predict(x)
    r2 = float(r2_score(y_actual, y_pred))
    mse = float(mean_squared_error(y_actual, y_pred))

    print("회귀계수(기울기):", float(model.coef_[0]))
    print("절편:", float(model.intercept_))
    print(f"Linear Regression R2: {r2:.6f}")
    print(f"Linear Regression MSE: {mse:.6f}")

    return {"r2": r2, "mse": mse, "y_actual": y_actual, "y_pred": y_pred, "skipped": None}


def plot_pca_explained_variance(explained_variance_ratio: np.ndarray) -> None:
    if len(explained_variance_ratio) == 0:
        print("PCA 설명분산비 이미지 생성을 건너뜁니다.")
        return

    output_path = IMAGE_DIR / "pca_explained_variance.png"
    try:
        components = [f"PC{i + 1}" for i in range(len(explained_variance_ratio))]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(components, explained_variance_ratio)
        ax.set_title("PCA Explained Variance Ratio")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_ylim(0, max(1.0, float(np.max(explained_variance_ratio)) * 1.15))
        for index, ratio in enumerate(explained_variance_ratio):
            ax.text(index, ratio, f"{ratio:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print("PCA 설명분산비 이미지 저장:", output_path)
    except Exception as exc:
        print(f"PCA 설명분산비 이미지 저장 실패, 분석은 계속합니다: {exc}")
        plt.close("all")


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    if correlation_matrix.empty:
        print("상관관계 히트맵 이미지 생성을 건너뜁니다.")
        return

    output_path = IMAGE_DIR / "correlation_heatmap.png"
    try:
        labels = correlation_matrix.columns.tolist()
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(correlation_matrix.values, vmin=-1, vmax=1)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_title("Pearson Correlation Heatmap")

        for row_index in range(len(labels)):
            for col_index in range(len(labels)):
                value = correlation_matrix.iloc[row_index, col_index]
                ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print("상관관계 히트맵 이미지 저장:", output_path)
    except Exception as exc:
        print(f"상관관계 히트맵 이미지 저장 실패, 분석은 계속합니다: {exc}")
        plt.close("all")


def plot_actual_vs_predicted(regression_result: dict) -> None:
    y_actual = regression_result.get("y_actual")
    y_pred = regression_result.get("y_pred")
    if y_actual is None or y_pred is None:
        print("실제 평점 vs 예측 평점 이미지 생성을 건너뜁니다.")
        return

    output_path = IMAGE_DIR / "actual_vs_predicted.png"
    try:
        y_actual = np.asarray(y_actual, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        valid_mask = np.isfinite(y_actual) & np.isfinite(y_pred)
        y_actual = y_actual[valid_mask]
        y_pred = y_pred[valid_mask]
        if len(y_actual) < 2:
            print("실제 평점 vs 예측 평점 이미지에 필요한 데이터가 부족합니다.")
            return

        min_value = float(min(np.min(y_actual), np.min(y_pred)))
        max_value = float(max(np.max(y_actual), np.max(y_pred)))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_actual, y_pred, alpha=0.65)
        ax.plot([min_value, max_value], [min_value, max_value])
        ax.set_title("Actual vs Predicted Rating")
        ax.set_xlabel("Actual Rating")
        ax.set_ylabel("Predicted Rating")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print("실제 평점 vs 예측 평점 이미지 저장:", output_path)
    except Exception as exc:
        print(f"실제 평점 vs 예측 평점 이미지 저장 실패, 분석은 계속합니다: {exc}")
        plt.close("all")


def save_analysis_summary(
    explained_variance_ratio: np.ndarray,
    regression_result: dict,
    arima_result: dict,
) -> None:
    lines = ["PCA Explained Variance Ratio:"]
    if len(explained_variance_ratio) == 0:
        lines.append("Skipped: PCA result is not available")
    else:
        for index, ratio in enumerate(explained_variance_ratio, start=1):
            lines.append(f"PC{index}: {ratio:.6f}")

    lines.append("")
    lines.append("Linear Regression:")
    if regression_result.get("skipped"):
        lines.append(f"Skipped: {regression_result['skipped']}")
    else:
        lines.append(f"R2: {regression_result['r2']:.6f}")
        lines.append(f"MSE: {regression_result['mse']:.6f}")

    lines.append("")
    lines.append("ARIMA:")
    if arima_result.get("skipped"):
        lines.append(f"Skipped: {arima_result['skipped']}")
    else:
        lines.append(f"Forecast: {arima_result['forecast']:.6f}")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("분석 요약 저장:", SUMMARY_PATH)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    try:
        df_raw = load_data()
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    print("원본 데이터 로드 완료, 행:", len(df_raw))

    df_clean = preprocess_missing_values(df_raw)
    df_clean = remove_outliers(df_clean)
    df_clean = remove_duplicates(df_clean)
    clean_path = OUTPUT_DIR / "hms_movies_clean.csv"
    df_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print("정리된 데이터 저장:", clean_path)

    df_scaled = normalize_data(df_clean)
    scaled_path = OUTPUT_DIR / "hms_movies_scaled.csv"
    df_scaled.to_csv(scaled_path, index=False, encoding="utf-8-sig")
    print("정규화된 데이터 저장:", scaled_path)

    df_pca, explained_variance_ratio = run_pca(df_scaled)
    pca_path = OUTPUT_DIR / "hms_movies_pca.csv"
    if df_pca.empty:
        print("PCA 결과가 없어 파일 저장을 건너뜁니다.")
    else:
        df_pca.to_csv(pca_path, index=False, encoding="utf-8-sig")
        print("PCA 결과 저장:", pca_path)

    correlation_matrix = analyze_correlation(df_scaled)
    arima_result = run_arima_forecast(df_clean)
    regression_result = run_linear_regression(df_scaled, df_pca)

    plot_pca_explained_variance(explained_variance_ratio)
    plot_correlation_heatmap(correlation_matrix)
    plot_actual_vs_predicted(regression_result)
    save_analysis_summary(explained_variance_ratio, regression_result, arima_result)


if __name__ == "__main__":
    main()

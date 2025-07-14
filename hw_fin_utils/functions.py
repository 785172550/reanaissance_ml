from scipy.stats import shapiro, normaltest, kstest
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    auc,
)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    recall_score,
    precision_score,
    f1_score,
)
import pandas_ta as ta
import yfinance as yf
import pandas as pd


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.

    Args:
        tickers (list): A list of stock ticker symbols.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical data for all tickers,
                          or an empty DataFrame if an error occurs.
    """
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker")
        print("Data download successful!")
        return data
    except Exception as e:
        print(f"Data download failed: {e}")
        return pd.DataFrame()


def check_numeric_distribution(data, col_name="numeric_column"):
    """
    数值型分布全面检查函数
    参数:
        data: Series
        col_name: for graph display
    """
    # 1. 基础检查
    print(f"\n=== {col_name} Distribution Check ===")
    print("Data count:", len(data))
    print("Missing values:", data.isna().sum())
    print("Descriptive stats:\n", data.describe())

    # 设置全局字体大小
    plt.rcParams.update({"font.size": 12})

    # 2. 可视化检查
    plt.figure(figsize=(15, 4))

    # 直方图+密度曲线
    plt.subplot(1, 3, 1)
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"{col_name} Histogram")

    # Q-Q
    plt.subplot(1, 3, 2)
    stats.probplot(data.dropna(), plot=plt)
    plt.title(f"{col_name} Q-Q Plot")

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(x=data)
    plt.title(f"{col_name} Boxplot")

    plt.tight_layout()
    plt.show()

    # 3. stats check for shapiro or normal
    clean_data = data.dropna()
    n = len(clean_data)

    print("\n【Normality Test】")
    try:
        if n < 50:
            stat, p = shapiro(clean_data)
            print(f"Shapiro-Wilk Test (n={n}<50): p={p:.4f}")
        else:
            stat, p = normaltest(clean_data)
            print(f"D'Agostino K² Test (n={n}≥50): p={p:.4f}")
    except Exception as e:
        print("normal check err", e)
        p = 0

    # kstest
    print("\n Other Distribution Tests")
    try:
        ks_stat, ks_p = kstest(
            (clean_data - clean_data.min()) / (clean_data.max() - clean_data.min()),
            "uniform",
        )
        print(f"Kolmogorov-Smirnov Uniformity Test: p={ks_p:.4f}")
    except Exception as e:
        print("K-S check err", e)

    # 5. result print
    alpha = 0.05
    print(f"\nDecision criterion (p={alpha}):")
    if p > alpha:
        print(f"→ Accept normality hypothesis (p={p:.4f} > {alpha})")
    else:
        print(f"→ Reject normality hypothesis (p={p:.4f} ≤ {alpha})")


def analyze_all_numeric_columns(df):
    """
    analyze distribution of all numeric columns in DataFrame
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    print(f"\n🔍 checked {len(numeric_cols)} count of{list(numeric_cols)}")

    for col in numeric_cols:
        try:
            check_numeric_distribution(df[col], col)
        except Exception as e:
            print(f"analyze err {col}: {e}")


def plot_confusion_matrix(
    y_true, y_pred, y_prob=None, label_names=None, model_name="Model"
):
    """
    draw confusion matrix
    """
    # 混淆矩阵
    confusion_mtx = confusion_matrix(y_true, y_pred)
    f, ax = plt.subplots(figsize=(4, 4), dpi=100)
    sns.heatmap(
        confusion_mtx,
        annot=True,
        linewidths=0.1,
        fmt=".0f",
        ax=ax,
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
    )

    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.title(f"{model_name} Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.show()


def draw_bar(acc, pre, recall):
    metrics = ["Accuracy", "Precision", "Recall"]
    values = [acc, pre, recall]  # 使用positive类的precision和recall，以及整体accuracy
    plt.figure(figsize=(5, 5))
    bars = plt.bar(metrics, values, color=["blue", "green", "red"])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.title("Classification Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)  # 设置y轴范围

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_auc(pred_nb, y_test):
    from sklearn.metrics import (
        auc,
        f1_score,
        precision_recall_curve,
        recall_score,
        roc_curve,
        precision_score,
    )

    fpr, tpr, _ = roc_curve(y_test, pred_nb)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, pred_nb)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.5f})".format(roc_auc))

    plt.plot(recall, precision, label="PR Curve (AUC = {:.5})".format(pr_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random", color="gray")

    # 设置坐标轴范围和标签
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate / Recall")
    plt.ylabel("True Positive Rate / Precision")
    plt.title("PR-ROC ")
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()


def visual_model_perf(y_test, y_pred, y_pred_label):
    acc = accuracy_score(y_test, y_pred_label)
    precision = precision_score(y_test, y_pred_label)  # 或 'micro', 'weighted'
    recall = recall_score(y_test, y_pred_label)
    f1 = f1_score(y_test, y_pred_label)

    draw_bar(acc, precision, recall)
    plot_confusion_matrix(
        y_test, y_pred_label, label_names=["Not Survived", "Survived"]
    )
    plot_auc(y_pred, y_test)


# rsi, macd, bbands, sma_50, sma_200
def calculate_technical_indicators(df):
    """
    Calculates technical indicators for a given stock data DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame with stock data (must contain 'Close').

    Returns:
        pandas.DataFrame: The DataFrame with added indicator columns.
    """
    print("Calculating technical indicators...")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)

    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df.dropna(inplace=True)
    print("Technical indicators calculated.")
    return df


def plot_price_and_mavg(df, ticker_name):
    """
    Generates and displays a line chart of closing prices and moving averages.

    Args:
        df (pandas.DataFrame): DataFrame with stock data and moving average columns.
        ticker_name (str): The name of the stock ticker for the chart title.
    """
    print(f"Generating line chart for {ticker_name}...")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(
        df.index,
        df["Close"],
        label=f"{ticker_name} Close Price",
        color="skyblue",
        linewidth=2,
    )
    ax.plot(
        df.index,
        df["SMA_50"],
        label="50-Day Moving Average",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        df.index,
        df["SMA_200"],
        label="200-Day Moving Average",
        color="red",
        linestyle="--",
    )

    ax.set_title(f"{ticker_name} Stock Price Analysis", fontsize=18)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(fontsize=10)
    fig.autofmt_xdate()
    plt.show()

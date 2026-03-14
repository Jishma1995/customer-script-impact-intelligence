from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def load_sentiment_model():
    """
    Load VADER sentiment analyzer.
    """
    return SentimentIntensityAnalyzer()


def calculate_sentiment_scores(comments, analyzer):
    """
    Calculate sentiment score for each comment.
    """
    scores = [analyzer.polarity_scores(c)["compound"] for c in comments]
    return scores


def label_sentiment(score):
    """
    Convert sentiment score into label.
    """
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"


def add_sentiment_labels(df: pd.DataFrame, score_col="sentiment_score"):
    """
    Add sentiment labels to dataframe.
    """
    df = df.copy()
    df["sentiment_label"] = df[score_col].apply(label_sentiment)
    return df
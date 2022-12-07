import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from project import analyze_sentiment, get_sentiment_score

def test_analyze_sentiment():
    # Test a positive phrase
    text = "I absolutely love Python! It's such a fun and versatile language."
    sentiment = analyze_sentiment(text)
    assert sentiment == 'positive'

    # Test a negative phrase
    text = "I absolutely hate Python! It's such a boring and inflexible language."
    sentiment = analyze_sentiment(text)
    assert sentiment == 'negative'

    # Test a neutral phrase
    text = "I am not sure how I feel about Python. It's an okay language, I guess."
    sentiment = analyze_sentiment(text)
    assert sentiment == 'neutral'


def test_get_sentiment_score():
    # Test a positive phrase
    text = "I absolutely love Python! It's such a fun and versatile language."
    sentiment_score = get_sentiment_score(text)
    assert sentiment_score > 0

    # Test a negative phrase
    text = "I absolutely hate Python! It's such a boring and inflexible language."
    sentiment_score = get_sentiment_score(text)
    assert sentiment_score < 0

    # Test a neutral phrase
    text = "I am not sure how I feel about Python. It's an okay language, I guess."
    sentiment_score = get_sentiment_score(text)
    assert 0 == pytest.approx(sentiment_score, abs=0.1)


def test_get_sentiment_score_using_vadersentiment():
    # Use VaderSentiment's sentiment analyzer to calculate the sentiment score
    # of the text, and compare the result to the output of the get_sentiment_score()
    # function from the project.py script
    analyzer = SentimentIntensityAnalyzer()
    text = "I absolutely love Python! It's such a fun and versatile language."
    scores = analyzer.polarity_scores(text)
    expected_sentiment_score = scores['compound']
    actual_sentiment_score = get_sentiment_score(text)
    assert expected_sentiment_score == actual_sentiment_score

    text = "I absolutely hate Python! It's such a boring and inflexible language."
    scores = analyzer.polarity_scores(text)
    expected_sentiment_score = scores['compound']
    actual_sentiment_score = get_sentiment_score(text)
    assert expected_sentiment_score == actual_sentiment_score

    text = "I am not sure how I feel about Python. It's an okay language, I guess."
    scores = analyzer.polarity_scores(text)
    expected_sentiment_score = scores['compound']
    actual_sentiment_score = get_sentiment_score(text)
    assert expected_sentiment_score == actual_sentiment_score

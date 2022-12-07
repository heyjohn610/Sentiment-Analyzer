from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    # Read the input text from the input.txt file
    text = read_text_from_file('input.txt')

    # Analyze the sentiment of the input text
    sentiment = analyze_sentiment(text)

    # Calculate the sentiment score of the input text
    sentiment_score = get_sentiment_score(text)

    # Output the sentiment and sentiment score to the terminal
    print(f"Sentiment: {sentiment}")
    print(f"Sentiment score: {sentiment_score:.2f}")


def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of the provided text using the VaderSentiment library.
    Returns 'positive', 'negative', or 'neutral'.
    """
    # Use VaderSentiment's sentiment analyzer to classify the sentiment of the text
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    # Classify the sentiment of the text based on the overall sentiment score
    if scores['compound'] > 0.1:
        return 'positive'
    elif scores['compound'] < -0.1:
        return 'negative'
    else:
        return 'neutral'


def get_sentiment_score(text: str) -> float:
    """
    Calculates the sentiment score of the provided text using the VaderSentiment library.
    Returns a float value in the range [-1.0, 1.0], where -1.0 indicates a very negative sentiment
    and 1.0 indicates a very positive sentiment.
    """
    # Use VaderSentiment to calculate the sentiment score of the text
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']


def read_text_from_file(filename: str) -> str:
    """
    Reads the text from the specified file and returns it as a string.
    """
    with open(filename, 'r') as f:
        return f.read()


if __name__ == "__main__":
    main()

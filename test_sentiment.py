import logging
from sentiment_analysis import analyze_sentiment

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

test_sentences = [
    {
        "text": "I just had the most amazing experience with your customer service team; they were so helpful, patient, and resolved my issue in no time, and I’m absolutely thrilled with the outcome!",
        "expected": "positive"
    },
    {
        "text": "I’ve been on hold for over an hour, the product I received was damaged, and the support team was rude and unhelpful, which has left me extremely frustrated and disappointed.",
        "expected": "negative"
    },
    {
        "text": "I’m calling to confirm my appointment for next week, and I just wanted to check if everything is set; there’s nothing urgent, just making sure we’re all on the same page.",
        "expected": "neutral"
    }
]

def test_english_sentiment():
    language = "en"
    print(f"\nTesting sentiment analysis for {language.upper()}:")
    for sentence in test_sentences:
        text = sentence["text"]
        expected = sentence["expected"]
        try:
            result = analyze_sentiment(text, language)
            print(f"Sentence: {text}")
            print(f"Sentiment: {result['label']} (score: {result['score']:.4f})")
            print(f"Expected: {expected}")
            print("-" * 80)
        except Exception as e:
            logger.error(f"Error analyzing '{text}': {str(e)}")
            print(f"Error analyzing '{text}': {str(e)}")

if __name__ == "__main__":
    test_english_sentiment()
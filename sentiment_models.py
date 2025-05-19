from transformers import pipeline
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

models = [
    "cardiffnlp/twitter-roberta-base-sentiment"
]

for model_name in models:
    try:
        logger.debug(f"Downloading model: {model_name}")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
        logger.debug(f"Successfully cached {model_name}")
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {str(e)}")
        raise

print("All sentiment analysis models downloaded and cached!")
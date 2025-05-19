import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDIC_MODEL_PATH = "indic_sentiment_model/final_model"
ENGLISH_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

SUPPORTED_LANGUAGES = {'en', 'hi', 'ta', 'te', 'ml', 'kn'}

try:
    english_sentiment_model = pipeline(
        "sentiment-analysis",
        model=ENGLISH_MODEL_NAME,
        tokenizer=ENGLISH_MODEL_NAME
    )
    logger.info("English sentiment analysis model loaded")
except Exception as e:
    logger.error(f"Failed to load English sentiment model: {str(e)}")
    raise

try:
    indic_tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL_PATH)
    indic_model = AutoModelForSequenceClassification.from_pretrained(INDIC_MODEL_PATH)
    indic_model.eval() 
    logger.info("Indic sentiment model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Indic sentiment model or tokenizer: {str(e)}")
    raise

def analyze_sentiment(text, language):
    """
    Analyze sentiment of the input text based on the language.
    Args:
        text (str): Input text to analyze.
        language (str): Language code ('en', 'hi', 'ta', 'te', 'ml', 'kn').
    Returns:
        dict: {'label': str, 'score': float} with sentiment label (lowercase) and confidence score.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided")
            return {"label": "unknown", "score": 0.0}
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
            return {"label": "unknown", "score": 0.0}

        if language == 'en':
            result = english_sentiment_model(text)[0]
            label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
            label = label_map[result['label']].lower()
            score = result['score']
            logger.debug(f"Text: {text[:50]}..., Language: {language}, Predicted: {label}, Score: {score:.4f}")
            return {"label": label, "score": float(score)}
        else:
            # Indic model (fine-tuned indic-bert or xlm-roberta-base)
            inputs = indic_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=256,  
                return_tensors="pt"
            )


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            indic_model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = indic_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence_score = probabilities[0, predicted_class].item()

            id2label = indic_model.config.id2label
            predicted_label = id2label[predicted_class].lower()  # Convert Negative->negative, etc.

            logger.debug(f"Text: {text[:50]}..., Language: {language}, Predicted: {predicted_label}, Score: {confidence_score:.4f}")

            return {
                "label": predicted_label,
                "score": float(confidence_score)
            }

    except Exception as e:
        logger.error(f"Sentiment analysis failed for text '{text[:50]}...': {str(e)}")
        return {"label": "unknown", "score": 0.0}
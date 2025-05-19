import logging
import requests
import os
from sentiment_analysis import analyze_sentiment

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:5000/transcribe"

INDIC_LANGUAGES = ['hi', 'ta', 'te', 'ml', 'kn']

TEST_TEXTS = {
    'hi': {
        'positive': [
            "यह एक शानदार उत्पाद है।",
            "मुझे इस सेवा से बहुत खुशी हुई।",
            "यह ऐप बहुत उपयोगी और तेज़ है।",
            "कंपनी का समर्थन शानदार है।",
            "मैं इसे सभी को सुझाऊंगा।"
        ],
        'negative': [
            "यह उत्पाद बहुत खराब है।",
            "सेवा बिल्कुल निराशाजनक थी।",
            "यह ऐप बार-बार क्रैश करता है।",
            "ग्राहक सेवा बहुत खराब है।",
            "मैं इसे किसी को नहीं सुझाऊंगा।"
        ]
    },
    'ta': {
        'positive': [
            "இது ஒரு சிறந்த தயாரிப்பு.",
            "இந்த சேவையில் நான் மிகவும் மகிழ்ச்சியடைந்தேன்.",
            "இந்த பயன்பாடு மிகவும் பயனுள்ளதாக உள்ளது.",
            "நிறுவனத்தின் ஆதரவு அற்புதமாக உள்ளது.",
            "நான் இதை எல்லோருக்கும் பரிந்துரைப்பேன்."
        ],
        'negative': [
            "இந்த தயாரிப்பு மிகவும் மோசமானது.",
            "சேவை மிகவும் ஏமாற்றமளிக்கிறது.",
            "இந்த பயன்பாடு தொடர்ந்து செயலிழக்கிறது.",
            "வாடிக்கையாளர் சேவை மோசமாக உள்ளது.",
            "நான் இதை யாருக்கும் பரிந்துரைக்க மாட்டேன்."
        ]
    },
    'te': {
        'positive': [
            "ఇది ఒక అద్భుతమైన ఉత్పత్తి.",
            "నేను ఈ సేవతో చాలా సంతోషించాను.",
            "ఈ యాప్ చాలా ఉపయోగకరంగా ఉంది.",
            "కంపెనీ మద్దతు అద్భుతంగా ఉంది.",
            "నేను దీన్ని అందరికీ సిఫార్సు చేస్తాను."
        ],
        'negative': [
            "ఈ ఉత్పత్తి చాలా చెడ్డది.",
            "సేవ చాలా నిరాశపరిచింది.",
            "ఈ యాప్ పదేపదే క్రాష్ అవుతోంది.",
            "కస్టమర్ సేవ చాలా దారుణంగా ఉంది.",
            "నేను దీన్ని ఎవరికీ సిఫార్సు చేయను."
        ]
    },
    'ml': {
        'positive': [
            "ഇത് ഒരു മികച്ച ഉൽപ്പന്നമാണ്.",
            "ഈ സേവനത്തിൽ ഞാൻ വളരെ സന്തുഷ്ടനാണ്.",
            "ഈ ആപ്പ് വളരെ ഉപയോഗപ്രദമാണ്.",
            "കമ്പനിയുടെ പിന്തുണ മികച്ചതാണ്.",
            "ഞാൻ ഇത് എല്ലാവർക്കും ശുപാർശ ചെയ്യും."
        ],
        'negative': [
            "ഈ ഉൽപ്പന്നം വളരെ മോശമാണ്.",
            "സേവനം വളരെ നിരാശാജനകമാണ്.",
            "ഈ ആപ്പ് പതിവായി ക്രാഷ് ചെയ്യുന്നു.",
            "ഉപഭോക്തൃ സേവനം വളരെ മോശമാണ്.",
            "ഞാൻ ഇത് ആർക്കും ശുപാർശ ചെയ്യില്ല."
        ]
    },
    'kn': {
        'positive': [
            "ಇದು ಒಂದು ಅದ್ಭುತವಾದ ಉತ್ಪನ್ನವಾಗಿದೆ.",
            "ನಾನು ಈ ಸೇವೆಯಿಂದ ತುಂಬಾ ಸಂತೋಷಗೊಂಡಿದ್ದೇನೆ.",
            "ಈ ಅಪ್ಲಿಕೇಶನ್ ತುಂಬಾ ಉಪಯುಕ್ತವಾಗಿದೆ.",
            "ಕಂಪನಿಯ ಬೆಂಬಲವು ಅದ್ಭುತವಾಗಿದೆ.",
            "ನಾನು ಇದನ್ನು ಎಲ್ಲರಿಗೂ ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ."
        ],
        'negative': [
            "ಈ ಉತ್ಪನ್ನವು ತುಂಬಾ ಕೆಟ್ಟದಾಗಿದೆ.",
            "ಸೇವೆಯು ತುಂಬಾ ನಿರಾಶಾದಾಯಕವಾಗಿದೆ.",
            "ಈ ಅಪ್ಲಿಕೇಶನ್ ಪದೇಪದೇ ಕ್ರ್ಯಾಶ್ ಆಗುತ್ತದೆ.",
            "ಗ್ರಾಹಕ ಸೇವೆಯು ತುಂಬಾ ಕಳಪೆಯಾಗಿದೆ.",
            "ನಾನು ಇದನ್ನು ಯಾರಿಗೂ ಶಿಫಾರಸು ಮಾಡುವುದಿಲ್ಲ."
        ]
    }
}

AUDIO_FILES = {
    'hi': "hindi_audio.wav",
    'ta': "tamil_audio.wav",
    'te': "telugu_audio.wav",
    'ml': "malayalam_audio.wav",
    'kn': "kannada_audio.wav"
}

EXPECTED_LABELS = {
    'positive': 'positive',
    'negative': 'negative'
}

def test_direct_sentiment():
    """Test sentiment_analysis.py directly with sample texts."""
    logger.info("Starting direct sentiment analysis tests")
    for lang in INDIC_LANGUAGES:
        logger.info(f"Testing language: {lang}")
        for sentiment, texts in TEST_TEXTS[lang].items():
            expected_label = EXPECTED_LABELS[sentiment]
            for text in texts:  
                try:
                    result = analyze_sentiment(text, lang)
                    logger.debug(f"Text: {text}, Language: {lang}, Result: {result}")
                    assert result['label'] == expected_label, f"Expected {expected_label}, got {result['label']} for {text}"
                    assert 0.0 <= result['score'] <= 1.0, f"Invalid score {result['score']} for {text}"
                    logger.info(f"PASS: {lang} {sentiment} - Text: {text}, Predicted: {result['label']}, Score: {result['score']:.4f}")
                except AssertionError as e:
                    logger.error(f"FAIL: {lang} {sentiment} - {str(e)}")
                except Exception as e:
                    logger.error(f"ERROR: {lang} {sentiment} - {str(e)}")

def test_api_sentiment():
    """Test Flask API with audio files (or skip if files are unavailable)."""
    logger.info("Starting Flask API sentiment analysis tests")
    for lang in INDIC_LANGUAGES:
        audio_file = AUDIO_FILES.get(lang)
        if not os.path.exists(audio_file):
            logger.warning(f"Audio file {audio_file} not found for {lang}. Skipping API test.")
            continue

        logger.info(f"Testing language: {lang} with audio: {audio_file}")
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': (audio_file, f, 'audio/wav')}
                data = {'language': lang}
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                logger.debug(f"API Response: {result}")

                assert 'transcription' in result, "Missing 'transcription' in response"
                assert 'language' in result, "Missing 'language' in response"
                assert result['language'] == lang, f"Expected language {lang}, got {result['language']}"
                assert 'sentiment' in result, "Missing 'sentiment' in response"
                assert result['sentiment']['label'] in ['positive', 'negative', 'unknown'], \
                    f"Invalid sentiment label: {result['sentiment']['label']}"
                assert 0.0 <= result['sentiment']['score'] <= 1.0, \
                    f"Invalid sentiment score: {result['sentiment']['score']}"

                logger.info(f"PASS: {lang} API - Transcription: {result['transcription'][:50]}..., "
                            f"Sentiment: {result['sentiment']['label']}, Score: {result['sentiment']['score']:.4f}")
        except AssertionError as e:
            logger.error(f"FAIL: {lang} API - {str(e)}")
        except Exception as e:
            logger.error(f"ERROR: {lang} API - {str(e)}")

def main():
    """Run all tests."""
    logger.info("Starting sentiment analysis tests")
    test_direct_sentiment()
    # test_api_sentiment()
    logger.info("Tests completed")

if __name__ == "__main__":
    main()
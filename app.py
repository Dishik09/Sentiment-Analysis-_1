from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from pydub import AudioSegment
import logging
import librosa
import soundfile as sf
import torch
import numpy as np
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from df.enhance import enhance, init_df, load_audio as df_load_audio
from datetime import datetime
from pymongo import MongoClient
import os
import re
import gridfs
from sentiment_analysis import analyze_sentiment

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mongo_client = MongoClient('mongodb://localhost:27017')
db = mongo_client['audio_transcriptions']
fs = gridfs.GridFS(db)
transcriptions_collection = db['transcriptions']

try:
    model = WhisperModel("large-v3", device="cpu", compute_type="float32")
    logger.debug("Faster-Whisper model loaded")
except Exception as e:
    logger.error(f"Failed to load whisper model: {str(e)}")
    raise

denoise_model = None
df_state = None
try:
    denoise_model, df_state, *_ = init_df()
    logger.debug("DeepFilterNet model loaded")
except Exception as e:
    logger.warning(f"Failed to load DeepFilterNet model: {str(e)}. Denoising functionality disabled.")

sarcasm_model = None
sarcasm_tokenizer = None
try:
    MODEL_PATH = "helinivan/english-sarcasm-detector"
    sarcasm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    logger.debug("Sarcasm detection model loaded")
except Exception as e:
    logger.warning(f"Failed to load sarcasm detection model: {str(e)}. Sarcasm detection disabled.")

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
SUPPORTED_LANGUAGES = {'en', 'ta', 'hi', 'kn', 'te', 'ml'}
LANGUAGE_PROMPTS = {
    "en": "The following is English text.",
    "ta": "இது தமிழ் உரை.",
    "hi": "यह हिंदी पाठ है।",
    "kn": "ಇದು ಕನ್ನಡ ಪಠ್ಯ.",
    "te": "ఇది తెలుగు పాఠ్యం.",
    "ml": "ഇത് മലയാളം ടെക്സ്റ്റ് ആണ്."
}
SCRIPT_RANGES = {
    'ta': r'[\u0B80-\u0BFF]',
    'hi': r'[\u0900-\u097F]',
    'kn': r'[\u0C80-\u0CFF]',
    'te': r'[\u0C00-\u0C7F]',
    'ml': r'[\u0D00-\u0D7F]'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_sarcasm_text(text):
    import string
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

def detect_sarcasm(text):
    if sarcasm_model is None or sarcasm_tokenizer is None:
        logger.error("Sarcasm detection model not loaded")
        return {"error": "Sarcasm detection unavailable: model not loaded"}
    try:
        processed_text = preprocess_sarcasm_text(text)
        logger.debug(f"Processed text for sarcasm detection: {processed_text}")
        tokenized_text = sarcasm_tokenizer(
            [processed_text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = sarcasm_model(**tokenized_text)
            probs = output.logits.softmax(dim=-1).tolist()[0]
        confidence = max(probs)
        prediction = probs.index(confidence)
        result = {
            "is_sarcastic": bool(prediction),
            "confidence": float(confidence)
        }
        logger.debug(f"Sarcasm detection result: {result}")
        return result
    except Exception as e:
        logger.error(f"Sarcasm detection failed: {str(e)}")
        return {"error": f"Sarcasm detection failed: {str(e)}"}

def load_audio(file_path):
    try:
        logger.debug(f"Loading {file_path} with pydub")
        audio = AudioSegment.from_file(file_path)
        logger.debug(f"Original audio: channels={audio.channels}, sample_rate={audio.frame_rate}, duration={audio.duration_seconds}s")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio = audio.normalize()
        logger.debug(f"Processed audio: channels={audio.channels}, sample_rate={audio.frame_rate}")
        temp_wav = "temp_audio.wav"
        audio.export(temp_wav, format="wav")
        logger.debug(f"Saved temporary WAV: {temp_wav}")
        return temp_wav
    except Exception as e:
        logger.error(f"Failed to load audio: {str(e)}")
        raise Exception(f"Failed to load audio: {str(e)}")

def denoise_audio(file_path):
    try:
        logger.debug(f"Denoising audio: {file_path}")
        result = df_load_audio(file_path)
        logger.debug(f"df_load_audio returned: {type(result)}, {len(result) if isinstance(result, tuple) else result}")
        if isinstance(result, tuple) and len(result) >= 2:
            audio, meta = result[:2]
            sr = meta.sample_rate if hasattr(meta, 'sample_rate') else meta
            logger.debug(f"Audio type: {type(audio)}, Sample rate: {sr}")
        else:
            raise ValueError(f"Unexpected return format from df_load_audio: {result}")
        if not isinstance(audio, torch.Tensor):
            raise ValueError(f"Audio data must be of type torch.Tensor, got {type(audio)}")
        expected_sr = df_state.sr()
        if sr != expected_sr:
            logger.warning(f"Sample rate {sr} does not match expected {expected_sr}. Resampling...")
            audio_np = audio.cpu().numpy()
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=expected_sr)
            audio = torch.from_numpy(audio_np).float()
            sr = expected_sr
            logger.debug(f"Resampled audio to sample rate {sr}")
        denoised_audio = enhance(denoise_model, df_state, audio)
        denoised_filename = "denoised_" + os.path.basename(file_path).rsplit('.', 1)[0] + ".wav"
        denoised_path = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename)
        if isinstance(denoised_audio, torch.Tensor):
            denoised_audio = denoised_audio.cpu().numpy()
        save_audio(denoised_path, denoised_audio, sr)
        logger.debug(f"Saved denoised audio: {denoised_path}")
        return denoised_path, denoised_filename
    except Exception as e:
        logger.error(f"Failed to denoise audio: {str(e)}")
        raise Exception(f"Failed to denoise audio: {str(e)}")

def validate_script(text, language):
    if not text or language == 'en':
        return True
    if language not in SCRIPT_RANGES:
        return True
    script_pattern = SCRIPT_RANGES[language]
    return bool(re.search(script_pattern, text))

def check_transcription_quality(text):
    pattern = r'(.)\1{4,}'
    return bool(re.search(pattern, text))

@app.route('/denoise', methods=['POST'])
def denoise_audio_endpoint():
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Request files: {request.files}")
    if denoise_model is None or df_state is None:
        logger.error("Denoising not available: DeepFilterNet model not loaded")
        return jsonify({'error': 'Denoising not available: DeepFilterNet model not loaded'}), 503
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        logger.error(f"Invalid file format: {file.filename}")
        return jsonify({'error': 'Invalid file format. Allowed: wav, mp3, m4a, flac'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        logger.debug(f"Saved file to {file_path}")
        denoised_path, denoised_filename = denoise_audio(file_path)
        return send_file(
            denoised_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=denoised_filename
        )
    except Exception as e:
        logger.error(f"Error denoising audio: {str(e)}")
        return jsonify({'error': f'Error denoising audio: {str(e)}'}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Removed uploaded file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {str(e)}")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Request form: {request.form}")
    logger.debug(f"Request files: {request.files}")
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        logger.error(f"Invalid file format: {file.filename}")
        return jsonify({'error': 'Invalid file format. Allowed: wav, mp3, m4a, flac'}), 400
    language = request.form.get('language')
    logger.debug(f"Received language parameter: {language}")
    if language and language not in SUPPORTED_LANGUAGES:
        logger.error(f"Invalid language: {language}")
        return jsonify({'error': f'Invalid language. Supported: {", ".join(SUPPORTED_LANGUAGES)}'}), 400
    strict_mode = request.form.get('strict_mode', 'false').lower() == 'true'
    logger.debug(f"Strict language mode: {strict_mode}")
    denoise = request.form.get('denoise', 'false').lower() == 'true'
    logger.debug(f"Denoise enabled: {denoise}")
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    temp_wav = None
    try:
        file.save(file_path)
        logger.debug(f"Saved file to {file_path}")
        with open(file_path, 'rb') as audio_file:
            audio_file_id = fs.put(audio_file, filename=filename)
        audio_path = file_path
        if denoise and denoise_model:
            audio_path, _ = denoise_audio(file_path)
            logger.debug(f"Using denoised audio: {audio_path}")
        temp_wav = load_audio(audio_path)
        initial_prompt = LANGUAGE_PROMPTS.get(language, "") if language else ""
        transcription_options = {
            "beam_size": 10,
            "temperature": 0.0,
            "best_of": 5,
            "task": "transcribe",
            "initial_prompt": initial_prompt
        }
        if language:
            transcription_options["language"] = language
            if strict_mode:
                transcription_options["suppress_language_detection"] = True
        segments, info = model.transcribe(temp_wav, **transcription_options)
        detected_language = info.language if language is None else language
        logger.debug(f"Detected language: {detected_language} (probability: {info.language_probability:.2f})")
        transcription_segments = []
        full_text = ""
        for segment in segments:
            full_text += segment.text
            transcription_segments.append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })
        full_text = full_text.strip()
        logger.debug(f"Transcription: {full_text}")
        warnings = []
        script_valid = validate_script(full_text, detected_language)
        if not script_valid:
            warnings.append(f"Transcription script does not match detected language {detected_language}")
            logger.warning(f"Script mismatch: language={detected_language}, transcription={full_text}")
            fallback_lang = 'te' if detected_language == 'ml' else 'ml'
            if fallback_lang in SUPPORTED_LANGUAGES:
                logger.debug(f"Retrying transcription with fallback language: {fallback_lang}")
                transcription_options["language"] = fallback_lang
                segments, info = model.transcribe(temp_wav, **transcription_options)
                full_text = "".join(segment.text for segment in segments).strip()
                detected_language = fallback_lang
                transcription_segments = [{"text": segment.text, "start": segment.start, "end": segment.end} for segment in segments]
                logger.debug(f"Fallback transcription: {full_text}")
                if validate_script(full_text, detected_language):
                    warnings.append(f"Fallback to {fallback_lang} succeeded")
                else:
                    warnings.append(f"Fallback to {fallback_lang} still produced incorrect script")
        if check_transcription_quality(full_text):
            warnings.append("Transcription contains repetitive characters, possibly due to audio noise")
        try:
            sentiment_result = analyze_sentiment(full_text, detected_language)
            logger.debug(f"Sentiment result: {sentiment_result}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            sentiment_result = {"label": "unknown", "score": 0.0}
            warnings.append(f"Sentiment analysis failed: {str(e)}")
        transcription_file = f"transcription_{filename}.txt"
        transcription_path = os.path.join(app.config['UPLOAD_FOLDER'], transcription_file)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        with open(transcription_path, 'rb') as trans_file:
            transcription_file_id = fs.put(trans_file, filename=transcription_file)
        response = {
            'transcription_file_id': str(transcription_file_id),
            'language': detected_language,
            'language_probability': float(info.language_probability),
            'segments': transcription_segments,
            'sentiment': sentiment_result,
            'audio_file_id': str(audio_file_id)
        }
        if warnings:
            response['warnings'] = warnings
        if denoise and denoise_model:
            response['denoised_file'] = os.path.basename(audio_path)
        document = {
            'uploaded_date': datetime.utcnow(),
            'language': detected_language,
            'transcription_file_id': str(transcription_file_id),
            'audio_file_id': str(audio_file_id),
            'response': response
        }
        transcriptions_collection.insert_one(document)
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Removed uploaded file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {str(e)}")
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                logger.debug(f"Removed temporary WAV: {temp_wav}")
            except Exception as e:
                logger.error(f"Failed to remove temporary WAV {temp_wav}: {str(e)}")
        if denoise and denoise_model and audio_path != file_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.debug(f"Removed denoised file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to remove denoised file {audio_path}: {str(e)}")
        if 'transcription_path' in locals() and os.path.exists(transcription_path):
            try:
                os.remove(transcription_path)
                logger.debug(f"Removed transcription file: {transcription_path}")
            except Exception as e:
                logger.error(f"Failed to remove transcription file {transcription_path}: {str(e)}")

@app.route('/sarcasm', methods=['POST'])
def sarcasm_detection():
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Request JSON: {request.json}")
    if not request.is_json:
        logger.error("No JSON data provided")
        return jsonify({'error': 'No JSON data provided'}), 400
    data = request.get_json()
    text = data.get('text')
    if not text or not isinstance(text, str):
        logger.error("Invalid or missing 'text' field")
        return jsonify({'error': "Invalid or missing 'text' field"}), 400
    try:
        result = detect_sarcasm(text)
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        return jsonify({'sarcasm': result}), 200
    except Exception as e:
        logger.error(f"Error processing sarcasm detection: {str(e)}")
        return jsonify({'error': f'Error processing sarcasm detection: {str(e)}'}), 500

@app.route('/transcriptions', methods=['GET'])
def get_transcriptions():
    try:
        transcriptions = list(transcriptions_collection.find().sort('uploaded_date', -1))
        for transcription in transcriptions:
            transcription['_id'] = str(transcription['_id'])
            transcription['uploaded_date'] = transcription['uploaded_date'].isoformat()
        return jsonify(transcriptions), 200
    except Exception as e:
        logger.error(f"Error retrieving transcriptions: {str(e)}")
        return jsonify({'error': f'Error retrieving transcriptions: {str(e)}'}), 500

@app.route('/transcription/<file_id>', methods=['GET'])
def get_transcription_file(file_id):
    try:
        file_data = fs.get(file_id)
        return send_file(
            file_data,
            mimetype='text/plain',
            as_attachment=True,
            download_name=file_data.filename
        )
    except Exception as e:
        logger.error(f"Error retrieving transcription file {file_id}: {str(e)}")
        return jsonify({'error': f'Error retrieving transcription file: {str(e)}'}), 500

@app.route('/audio/<file_id>', methods=['GET'])
def get_audio_file(file_id):
    try:
        file_data = fs.get(file_id)
        return send_file(
            file_data,
            mimetype='audio/' + file_data.filename.rsplit('.', 1)[1].lower(),
            as_attachment=True,
            download_name=file_data.filename
        )
    except Exception as e:
        logger.error(f"Error retrieving audio file {file_id}: {str(e)}")
        return jsonify({'error': f'Error retrieving audio file: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model': 'large-v3',
        'denoise_model': 'DeepFilterNet' if denoise_model else 'None'
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
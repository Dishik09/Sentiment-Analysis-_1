from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model_path = "./whisper_model"
files = ["uploads/common_voice_ta_24035574.mp3"]

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

for file_path in files:
    logger.debug(f"Loading {file_path}")
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (1 << (8 * audio.sample_width - 1))
    logger.debug(f"Samples shape: {samples.shape}")

    inputs = processor.feature_extractor(
        samples,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features
    logger.debug(f"Feature extractor output shape: {inputs.shape}")

    try:
        predicted_ids = model.generate(
            inputs,
            max_length=1000
        )
        logger.debug(f"Predicted IDs shape: {predicted_ids.shape}")
        logger.debug(f"Predicted IDs: {predicted_ids.tolist()}")
        if predicted_ids.size(1) <= 1:
            logger.warning(f"Empty transcription for {file_path}")
            continue
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.debug(f"Transcription for {file_path}: {transcription}")
    except Exception as e:
        logger.error(f"Transcription failed for {file_path}: {str(e)}")
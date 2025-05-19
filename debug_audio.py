import torchaudio
import pydub
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

mp3_file = "uploads/common_voice_ta_24035574.mp3"

try:
    logger.debug("Testing torchaudio.load")
    waveform, sample_rate = torchaudio.load(mp3_file)
    logger.debug(f"torchaudio: sample_rate={sample_rate}, shape={waveform.shape}")
except Exception as e:
    logger.error(f"torchaudio failed: {str(e)}")

try:
    logger.debug("Testing pydub")
    audio = pydub.AudioSegment.from_file(mp3_file)
    temp_wav = "test_pydub_output.wav"
    audio.export(temp_wav, format="wav")
    logger.debug(f"pydub exported to {temp_wav}")

    waveform, sample_rate = torchaudio.load(temp_wav)
    logger.debug(f"torchaudio loaded WAV: sample_rate={sample_rate}, shape={waveform.shape}")
    os.remove(temp_wav)
except Exception as e:
    logger.error(f"pydub failed: {str(e)}")
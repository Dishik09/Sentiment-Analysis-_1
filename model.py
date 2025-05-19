from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.DEBUG)
model = WhisperModel("large-v3", device="cpu", compute_type="float32")
print("Model downloaded and cached!")
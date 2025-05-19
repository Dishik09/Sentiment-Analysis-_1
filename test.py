from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_path = "./whisper_model"
try:
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    print("Model and processor loaded successfully!")
except Exception as e:
    print(f"Error: {str(e)}")
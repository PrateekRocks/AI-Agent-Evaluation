import os
from dotenv import load_dotenv
from deepeval.models import GeminiModel

load_dotenv()

model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
api_key = os.getenv("GOOGLE_API_KEY")

eval_model = GeminiModel(
    model=model_name,
    api_key=api_key
)

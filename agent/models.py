import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.3,
    streaming=True
)

# response = model.invoke("tell me something about the sun in 2 sentences")
# print(response.text)
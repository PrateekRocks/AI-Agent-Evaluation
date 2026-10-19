from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3,
    streaming=True
)

# response = model.invoke("tell me something about the sun in 2 sentences")
# print(response.text)
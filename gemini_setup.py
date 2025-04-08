from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

def get_LLM() -> ChatGoogleGenerativeAI:

    if "GOOGLE_API_KEY" not in os.environ:
        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
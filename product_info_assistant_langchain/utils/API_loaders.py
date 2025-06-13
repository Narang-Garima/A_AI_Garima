import os 
from dotenv import load_dotenv

def load_api_keys():
    load_dotenv(override=True)

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
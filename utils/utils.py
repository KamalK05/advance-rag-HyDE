# utils/utils.py
import os
import yaml
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_llm(config):
    provider = config["llm"]["provider"]
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("'OPENAI_API_KEY' is not present in .env")
        return ChatOpenAI(
           api_key = api_key,
           model = config["llm"]["model_openai"],
           temperature = config["llm"].get("temperature", 0.3),
           max_tokens = config["llm"].get("max_tokens", 1000),
        )
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("'GOOGLE_API_KEY' is not present in .env")
        return ChatGoogleGenerativeAI(
            google_api_key = api_key,
            model = config["llm"]["model_gemini"],
            temperature = config["llm"].get("temperature", 0.3),
            max_output_tokens = config["llm"].get("max_tokens", 1000),
        )
    else:
        raise ValueError("Provider must be one of: openai | gemini")

def get_embedding_model(config):
       provider = config["llm"]["provider"]

       if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("'OPENAI_API_KEY' is not present in .env")
            return OpenAIEmbeddings(
                api_key = api_key,
                model = config["embedding"].get("openai_model", "text-embedding-3-small")
            )
       elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("'GOOGLE_API_KEY' is not present in .env")
            return GoogleGenerativeAIEmbeddings(
                google_api_key = api_key,
                model = config["embedding"].get("gemini_model", "models/text-embedding-004")
            )
       else:
            raise ValueError("Provider must be one of: openai | gemini")

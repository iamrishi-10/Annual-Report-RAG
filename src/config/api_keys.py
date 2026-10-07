import os

from dotenv import load_dotenv

from src.config.settings import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def is_key_set(env_var: str) -> bool:
    return bool(os.getenv(env_var))


def check_api_keys() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": is_key_set("OPENAI_API_KEY"),
        "PINECONE_API_KEY": is_key_set("PINECONE_API_KEY"),
        "COHERE_API_KEY": is_key_set("COHERE_API_KEY"),
    }


if __name__ == "__main__":
    for key, is_set in check_api_keys().items():
        print(f"{key}: {'SET' if is_set else 'NOT SET'} starting from {os.getenv(key)[:8]}...")

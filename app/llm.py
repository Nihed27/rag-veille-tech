import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    """Renvoie le LLM appelé via OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


if __name__ == "__main__":
    reponse = get_llm().invoke("Réponds en une phrase : qu'est-ce qu'un LLM ?")
    print(reponse.content)
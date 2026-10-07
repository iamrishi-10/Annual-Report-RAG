"""
Standalone LangSmith tracing smoke test.

Runs one small LangChain LLM call (prompt -> model -> parser) and confirms
it appears as a trace in LangSmith. Kept separate from the main RAG
pipeline so it can be run in isolation to verify tracing setup.

Requires these vars in .env:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your langsmith api key>
    LANGCHAIN_PROJECT=<your langsmith project name>
    OPENAI_API_KEY=<your openai api key>
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import GENERATION_MODEL, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    model = ChatOpenAI(model=GENERATION_MODEL)
    prompt = ChatPromptTemplate.from_messages(
        [("human", "{question}")]
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    question = "Explain LangSmith tracing in 2 lines."
    answer = chain.invoke({"question": question})

    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()

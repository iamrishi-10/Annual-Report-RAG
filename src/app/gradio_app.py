"""
Gradio demo UI for the permanent RAG chain (similarity -> MultiQuery ->
Cohere rerank -> answer generation).

This module only wires a question box to stream_rag_query() and displays
the streamed result. It does not retrieve documents, rerank, build context,
or call the LLM directly - all of that stays inside
src/generation/rag_chain.py. No chat history, latency/cost tracking, or
advanced settings yet - this is the first, minimal demo-ready UI.

Run with: python -m src.app.gradio_app
"""

from collections.abc import Iterator

from dotenv import load_dotenv
import gradio as gr

from src.config.settings import PROJECT_ROOT
from src.generation.rag_chain import stream_rag_query
from src.utils.logger import get_logger


logger = get_logger(__name__)

SOURCE_HEADERS = ["Source", "Page", "Chunk ID"]


def answer_question(question: str) -> Iterator[tuple[str, list[list]]]:
    """Stream the RAG chain's answer for one question, updating the UI as it goes.

    Args:
        question: The user's question, as typed into the UI.

    Yields:
        (answer_text, source_rows) tuples, mirroring stream_rag_query():
        sources appear as soon as retrieval/reranking finishes, then the
        answer text grows with each yield. source_rows is a list of
        [source_number, page_number, chunk_id] rows, one per final,
        reranked source document. On a blank question or an internal
        failure, yields once with a message in place of an answer and no
        source rows, rather than raising into the UI.
    """
    if not question or not question.strip():
        yield "Please enter a question.", []
        return

    try:
        logger.info("UI question: %s", question)

        yield from stream_rag_query(question)

    except Exception as error:
        logger.exception("Failed to answer question in UI")
        yield f"Something went wrong while answering: {error}", []


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks app: question in, answer + sources out."""
    with gr.Blocks(title="Annual Report Assistant") as demo:
        gr.Markdown(
            "# Annual Report Assistant\n"
            "Ask a question about Infosys' Integrated Annual Report 2024-25."
        )

        question_input = gr.Textbox(
            label="Question",
            placeholder="e.g. What was Infosys' total revenue in fiscal 2025?",
        )
        ask_button = gr.Button("Ask", variant="primary")

        answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)
        sources_output = gr.Dataframe(
            headers=SOURCE_HEADERS,
            label="Sources",
            interactive=False,
        )

        ask_button.click(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )

    return demo


def main() -> None:
    """Load .env, build the app, and launch it."""
    load_dotenv(PROJECT_ROOT / ".env")

    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()

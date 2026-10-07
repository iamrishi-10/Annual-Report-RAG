"""
Helper functions for evaluating retrieval ranking quality.

This module scores how good the retrieval ranking was for a set of final
source pages against a golden question's expected source pages. It does
not run the RAG chain, retrieval, reranking, call the LLM, generate
answers, or save JSON.
"""


def normalize_pages(pages: list[int | float]) -> list[int]:
    """Convert a list of page numbers to plain ints.

    Actual source pages (e.g. from Pinecone metadata) may come back as
    floats (15.0), while expected_source_pages are plain ints (15).

    Args:
        pages: Page numbers, possibly floats.

    Returns:
        The same page numbers, each converted with int().
    """
    return [int(page) for page in pages]


def calculate_retrieval_metrics(
    actual_source_pages: list[int | float],
    expected_source_pages: list[int],
) -> dict:
    """Score the ranking quality of the final retrieved source pages.

    Args:
        actual_source_pages: Page numbers from the RAG chain's final
            sources, in rank order (index 0 is the top-ranked source).
        expected_source_pages: Page numbers the golden question expects.

    Returns:
        A dict with:
            retrieval_hit: True if at least one expected page appears in
                actual_source_pages.
            precision_at_k: Fraction of actual_source_pages that are an
                expected page (0.0 if actual_source_pages is empty).
            recall_at_k: Fraction of expected_source_pages that were
                found in actual_source_pages (0.0 if expected_source_pages
                is empty).
            first_relevant_rank: 1-indexed rank of the first actual page
                that is an expected page, or None if there is no hit.
            mrr: 1 / first_relevant_rank, or 0.0 if there is no hit.
    """
    normalized_actual_pages = normalize_pages(actual_source_pages)
    normalized_expected_pages = set(normalize_pages(expected_source_pages))

    if not normalized_actual_pages:
        precision_at_k = 0.0
    else:
        relevant_count = sum(
            1 for page in normalized_actual_pages if page in normalized_expected_pages
        )
        precision_at_k = relevant_count / len(normalized_actual_pages)

    if not normalized_expected_pages:
        recall_at_k = 0.0
    else:
        matched_expected_pages = normalized_expected_pages.intersection(
            normalized_actual_pages
        )
        recall_at_k = len(matched_expected_pages) / len(normalized_expected_pages)

    first_relevant_rank = None
    for rank, page in enumerate(normalized_actual_pages, start=1):
        if page in normalized_expected_pages:
            first_relevant_rank = rank
            break

    retrieval_hit = first_relevant_rank is not None
    mrr = 1 / first_relevant_rank if retrieval_hit else 0.0

    return {
        "retrieval_hit": retrieval_hit,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "first_relevant_rank": first_relevant_rank,
        "mrr": mrr,
    }

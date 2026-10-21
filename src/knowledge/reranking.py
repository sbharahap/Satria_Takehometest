from src.models import Source


def rerank(query: str, candidates: list[Source], limit: int) -> list[Source]:
    """Lexical reranking within eligible dense candidates, not a second global index.

    Character n-grams retain Indonesian word-form overlap without a stemmer/model download.
    The stored score remains the original dense cosine similarity for thresholding/display.
    """
    if len(candidates) <= 1:
        return candidates[:limit]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        matrix = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        ).fit_transform([query, *[source.text for source in candidates]])
    except ValueError:
        return candidates[:limit]
    lexical = cosine_similarity(matrix[:1], matrix[1:]).ravel()
    ranked = sorted(
        zip(candidates, lexical, strict=True),
        key=lambda item: (float(item[1]), item[0].score),
        reverse=True,
    )
    return [source for source, _ in ranked[:limit]]

import time
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily. 
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("Initializing FlashRank model for local reranking")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception as e:
            _ranker = Ranker()
    return _ranker



def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """

    if not documents:
        logfire.warning("No documents provided for reranking.")
        return []
    
    logfire.info(f"Reranking {len(documents)} documents for query")
    start_time = time.time()

    try:

        # 1. Setup
        ranker = _get_ranker()
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        docs_with_ids = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        # 2. Prepare Request for FlashRank
        # We use RerankRequest(query, passages) which returns the docs in ranked order based on attention/relevency score
        request = RerankRequest(
            query=query,
            passages=docs_with_ids
        )

        # 3. Execute Reranking
        results = ranker.rerank(request)

        # 4. Extract Sorted Documents
        # FlashRank returns results sorted by relevance score (highest first)
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        rerank_time = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"[Reranker] Done in {rerank_time:.2f}s. Top semantic score: {top_score}")
        
        return reranked_docs
    except Exception as e:
        logfire.error(f"[Reranker] FlashRank reranking failed: {e}")
        # Fallback: return original documents if reranking fails
        return documents[:top_n]
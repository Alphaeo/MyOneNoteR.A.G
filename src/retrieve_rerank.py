"""
Étape 5 — Récupération HYBRIDE (dense + sparse, fusion RRF côté Qdrant) +
reranking.

Fonctions réutilisées par generate.py. Pas de CLI ici.

Flux :
    question -> embed_dense(question) + embed_sparse(question)
             -> Qdrant : prefetch dense (dense_top_k) + prefetch sparse (sparse_top_k)
                         -> fusion RRF -> fused_top_k candidats
             -> reranker cross-encoder -> top_k_final chunks
"""
from qdrant_client import models
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# TODO 1 — Encoder la question avec les MÊMES modèles que pour les documents
#   (embed_index.py) :
#   - dense : appliquer le préfixe/instruction "query" si Qwen3-Embedding en
#     a besoin (voir note dans embed_index.py)
#   - sparse : même modèle fastembed (BM25) que pour l'indexation


def embed_query_dense(question: str, dense_model) -> list[float]:
    return dense_model.encode(question, prompt_name="query", normalize_embeddings=True).tolist()

def embed_query_sparse(question: str, sparse_model):
    return next(sparse_model.embed([question]))


# TODO 2 — Requête hybride sur Qdrant avec prefetch + fusion.
#   from qdrant_client import models
#   client.query_points(
#       collection_name,
#       prefetch=[
#           models.Prefetch(query=dense_vec, using="dense", limit=config["retrieval"]["dense_top_k"]),
#           models.Prefetch(query=sparse_vec, using="sparse", limit=config["retrieval"]["sparse_top_k"]),
#       ],
#       query=models.FusionQuery(fusion=models.Fusion.RRF),
#       limit=config["retrieval"]["fused_top_k"],
#       with_payload=True,
#   )
#   Si `retrieval.hybrid` est false dans la config, faire un simple
#   query_points avec query=dense_vec, using="dense" (pas de prefetch/fusion).
#
#   Doc : https://qdrant.tech/documentation/concepts/hybrid-queries/


def hybrid_search(dense_vec, sparse_vec, config: dict) -> list[dict]:
    client = QdrantClient(path=config["vector_store"]["persist_dir"])
    collection_name = config["vector_store"]["collection_name"]
    if config["retrieval"]["hybrid"]:
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=config["retrieval"]["dense_top_k"],
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using="sparse",
                limit=config["retrieval"]["sparse_top_k"],
            ),
        ]
        fusion_query = models.FusionQuery(fusion=models.Fusion.RRF)
        response = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=fusion_query,
            limit=config["retrieval"]["fused_top_k"],
            with_payload=True,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=dense_vec,
            using="dense",
            limit=config["retrieval"]["fused_top_k"],
            with_payload=True,
        )
    client.close()
    candidates = []
    for point in response.points:
        payload = point.payload
        candidates.append({
            "page_id": payload["page_id"],
            "page_title": payload["page_title"],
            "source_url": payload["source_url"],
            "text": payload["text"],
            "score": point.score,
        })
    return candidates


# TODO 3 — Reranking des candidats.
#   Option simple (bge-reranker-v2-m3) :
#     from sentence_transformers import CrossEncoder
#     scores = CrossEncoder(model_name).predict([(question, chunk_text), ...])
#   Option Qwen3-Reranker : suivre le prompt template spécifique documenté
#   sur https://huggingface.co/Qwen/Qwen3-Reranker-0.6B (scoring par
#   probabilité du token "yes").
#   Trier les candidats par score décroissant et garder top_k_final.


def rerank(question: str, candidates: list[dict], top_k_final: int, cross_encoder) -> list[dict]:
    scores = cross_encoder.predict([(question, c["text"]) for c in candidates])
    for candidate, score in zip(candidates, scores):
        candidate["score"] = score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k_final]


# TODO 4 — Fonction de haut niveau combinant : embed_query_dense +
#   embed_query_sparse -> hybrid_search -> rerank. À appeler depuis
#   generate.py ET depuis evaluate_ragas.py (l'éval a besoin des `contexts`
#   retournés ici pour calculer context_precision/context_recall).


def retrieve(question: str, config: dict, dense_model, sparse_model, cross_encoder) -> list[dict]:
    """Retourne les top_k_final chunks les plus pertinents, rerankés,
    prêts à être injectés dans le prompt de génération."""
    dense_vec = embed_query_dense(question, dense_model)
    sparse_vec = embed_query_sparse(question, sparse_model)
    candidates = hybrid_search(dense_vec, sparse_vec, config)
    top_k_final = config["retrieval"]["top_k_final"]
    return rerank(question, candidates, top_k_final, cross_encoder)

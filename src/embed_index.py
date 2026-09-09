"""
Étape 4 — Génération des embeddings DENSE + SPARSE et indexation hybride
dans Qdrant (vecteurs nommés).

INCRÉMENTAL : ne ré-embedder que les pages ajoutées/modifiées, et supprimer
de Qdrant les points des pages modifiées/supprimées avant de les réinsérer
— voir data/raw/last_sync.json et ARCHITECTURE.md section "Synchronisation
incrémentale". Chaque point Qdrant doit avoir `page_id` dans son payload
pour permettre ce ciblage par filtre.

Entrée  : data/processed/chunks/<page_id>.jsonl (un fichier par page, voir chunk.py)
Sortie  : data/index/ (collection Qdrant persistée en mode embarqué, avec
          un vecteur nommé "dense" et un vecteur nommé "sparse" par point)

Points d'attention :
    - Qwen3-Embedding utilise potentiellement des préfixes/instructions
      différents pour indexer un document vs pour encoder une requête de
      recherche. Vérifie la carte du modèle sur Hugging Face
      (https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) pour le format exact
      avant d'indexer quoi que ce soit — sinon il faudra tout ré-embedder.
    - Normaliser les vecteurs dense (cosine similarity) si le modèle le
      recommande.
    - Le vecteur sparse (BM25 via fastembed) n'a PAS besoin d'un modèle de
      langage — c'est un scoring statistique sur les tokens du texte. Pas de
      normalisation nécessaire, mais garder le MÊME tokenizer/modèle BM25
      entre indexation et requête.
"""

import yaml
import json
import os
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (VectorParams, SparseVectorParams, Distance, PointStruct, SparseVector)
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
import logging
from qdrant_client import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO 1 — Charger config.yaml (sections "embedding", "vector_store", "retrieval")

def load_config() -> dict:
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# TODO 2 — Instancier le modèle d'embedding DENSE selon le backend choisi :
#   - "sentence_transformers" : SentenceTransformer(model_name)
#   - "ollama" : client HTTP vers OLLAMA_BASE_URL + /api/embeddings
#   Retourner une fonction unifiée `embed_dense(texts: list[str]) -> list[list[float]]`


def load_dense_embedding_model(config: dict):
    if config["embedding"]["dense"]["backend"] == "sentence_transformers":
        model_name = config["embedding"]["dense"]["model_name"]
        model = SentenceTransformer(model_name)
        # batch_size réduit : la machine a très peu de RAM libre, de gros
        # batches provoquent du swap disque et des ralentissements énormes
        # et erratiques (constaté : certains batches ont pris 30+ minutes).
        return lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=4).tolist()
    elif config["embedding"]["dense"]["backend"] == "ollama":
        ollama_url = config["embedding"]["dense"]["ollama_url"]
        headers = {"Authorization": f"Bearer {config['embedding']['dense']['ollama_token']}"}
        return lambda texts: [
            requests.post(f"{ollama_url}/api/embeddings", headers=headers, json={"texts": texts}).json()
        ]
    return None


# TODO 3 — Instancier le modèle d'embedding SPARSE :
#   from fastembed import SparseTextEmbedding
#   model = SparseTextEmbedding(model_name=config["embedding"]["sparse"]["model_name"])  # ex: "Qdrant/bm25"
#   model.embed(texts) -> génère des SparseEmbedding(indices=[...], values=[...])
#   Retourner une fonction unifiée `embed_sparse(texts: list[str]) -> list[SparseEmbedding]`
#
#   Alternative si tu veux du sparse "appris" (meilleure qualité, plus lourd) :
#   un modèle SPLADE via fastembed également (voir leur liste de modèles
#   supportés) — même interface, juste changer model_name.


def load_sparse_embedding_model(config: dict):
    model_name = config["embedding"]["sparse"]["model_name"]
    model = SparseTextEmbedding(model_name=model_name)
    return lambda texts: model.embed(texts)
    


# TODO 4 — Lire les chunks d'UNE page depuis data/processed/chunks/<page_id>.jsonl


def load_chunks_for_page(page_id: str) -> list[dict]:
    chunks = []
    with open(f"data/processed/chunks/{page_id}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


# TODO 5 — Batcher les chunks et calculer les embeddings dense ET sparse
#   (attention à la taille de batch selon ta RAM/VRAM disponible — le dense
#   est la partie coûteuse, le sparse/BM25 est quasi gratuit en comparaison)


def embed_chunks(chunks: list[dict], embed_dense_fn, embed_sparse_fn) -> tuple[list, list]:
    """Retourne (dense_vectors, sparse_vectors), alignés sur l'ordre de `chunks`."""
    texts = [chunk["text"] for chunk in chunks]
    dense_vectors = embed_dense_fn(texts)
    sparse_vectors = list(embed_sparse_fn(texts))
    return dense_vectors, sparse_vectors


# TODO 6 — Créer/ouvrir la collection Qdrant avec DEUX vecteurs nommés :
#   - Mode embarqué (recommandé pour démarrer, pas de Docker) :
#       QdrantClient(path=config["vector_store"]["persist_dir"])
#   - Mode serveur (si tu préfères Docker) :
#       QdrantClient(host=..., port=...)
#   Puis, si la collection n'existe pas déjà (client.collection_exists(...)) :
#   - client.create_collection(
#       collection_name,
#       vectors_config={
#           "dense": VectorParams(size=<dim du modèle dense>, distance=Distance.COSINE),
#       },
#       sparse_vectors_config={
#           "sparse": SparseVectorParams(),
#       },
#     )
#   - client.upsert(collection_name, points=[PointStruct(
#       id=...,
#       vector={"dense": dense_vec, "sparse": SparseVector(indices=..., values=...)},
#       payload={"text": ..., "page_title": ..., "source_url": ..., ...},
#     )])
#     Note : les IDs Qdrant doivent être des int ou des UUID — si tes
#     chunk_id sont des strings arbitraires, convertis-les en UUID5
#     déterministe à partir du chunk_id (pour pouvoir réindexer de façon
#     idempotente).
#
#   Doc Qdrant hybrid queries : https://qdrant.tech/documentation/concepts/hybrid-queries/
#   Doc Qdrant Python client  : https://python-client.qdrant.tech/


def index_chunks(chunks: list[dict], dense_vectors: list, sparse_vectors: list, config: dict, qdrant_client: QdrantClient) -> None:
    collection_name = config["vector_store"]["collection_name"]
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name,
            vectors_config={
                "dense": VectorParams(size=len(dense_vectors[0]), distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
            vector={"dense": dense_vec, "sparse": SparseVector(indices=sparse_vec.indices, values=sparse_vec.values)},
            payload={
                "text": chunk["text"],
                "page_id": chunk["page_id"],
                "page_title": chunk["page_title"],
                "source_url": chunk["source_url"],
            },
        )
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors)
    ]
    qdrant_client.upsert(collection_name, points=points)

# TODO 7 — Supprimer de Qdrant tous les points appartenant à une page donnée
#   (utilisé pour les pages "modified" avant réinsertion, et pour les pages
#   "deleted") :
#   client.delete(
#       collection_name,
#       points_selector=models.FilterSelector(
#           filter=models.Filter(must=[models.FieldCondition(
#               key="page_id", match=models.MatchValue(value=page_id))])
#       ),
#   )


def delete_page_from_index(page_id: str, config: dict, qdrant_client: QdrantClient) -> None:
    collection_name = config["vector_store"]["collection_name"]
    qdrant_client.delete(
        collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(
                key="page_id", match=models.MatchValue(value=page_id))])
        ),
    )


# TODO 8 — Orchestration incrémentale :
#   1. Lire data/raw/last_sync.json
#   2. Pour chaque page_id dans modified + deleted : delete_page_from_index(page_id)
#   3. Pour chaque page_id dans added + modified :
#      - load_chunks_for_page(page_id)
#      - embed_chunks() puis index_chunks() (upsert — pas besoin de delete
#        d'abord pour "added" puisque rien n'existait)
#   4. Si last_sync.json absent (premier run / mode --full) : traiter TOUTES
#      les pages présentes dans data/processed/chunks/*.jsonl comme "added"


def load_last_sync(path: str = "data/raw/last_sync.json") -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config = load_config()
    embed_dense = load_dense_embedding_model(config)
    embed_sparse = load_sparse_embedding_model(config)

    last_sync = load_last_sync()
    if last_sync is None:
        # Premier run / mode --full : traiter toutes les pages comme "added"
        page_ids = [
            fname[: -len(".jsonl")]
            for fname in os.listdir("data/processed/chunks")
            if fname.endswith(".jsonl")
        ]
        last_sync = {"added": page_ids, "modified": [], "deleted": []}

    qdrant_client = QdrantClient(path=config["vector_store"]["persist_dir"])

    for page_id in last_sync["modified"] + last_sync["deleted"]:
        delete_page_from_index(page_id, config, qdrant_client)

    for page_id in last_sync["added"] + last_sync["modified"]:
        chunks = load_chunks_for_page(page_id)
        dense_vectors, sparse_vectors = embed_chunks(chunks, embed_dense, embed_sparse)
        index_chunks(chunks, dense_vectors, sparse_vectors, config, qdrant_client)

    qdrant_client.close()
    print(f"{len(last_sync['added'])} pages indexées, {len(last_sync['modified'])} mises à jour, {len(last_sync['deleted'])} supprimées")


if __name__ == "__main__":
    main()

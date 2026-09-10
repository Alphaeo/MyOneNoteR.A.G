"""
Interface Gradio locale pour poser des questions à ton RAG OneNote.

Enveloppe simplement retrieve_rerank.retrieve() + generate.generate_answer()
dans un gr.ChatInterface. Les modèles sont chargés une seule fois au
démarrage du script (pas à chaque question).

Lancement : python src/app.py, puis ouvre l'URL locale affichée dans le
terminal (http://127.0.0.1:7860 par défaut).
"""
import gradio as gr

from generate import (
    load_config,
    build_system_prompt,
    build_user_message,
    generate_answer,
    load_generation_model,
)
from retrieve_rerank import retrieve
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

print("Chargement des modèles (peut prendre un moment)...")
config = load_config()
dense_model = SentenceTransformer(config["embedding"]["dense"]["model_name"])
sparse_model = SparseTextEmbedding(model_name=config["embedding"]["sparse"]["model_name"])
cross_encoder = CrossEncoder(config["reranker"]["model_name"], model_kwargs={"dtype": "bfloat16"})
model_name = load_generation_model(config)
system_prompt = build_system_prompt()
print("Modèles chargés, interface prête.")


def respond(message: str, history: list[dict]) -> str:
    chunks = retrieve(message, config, dense_model, sparse_model, cross_encoder)
    user_message = build_user_message(message, chunks, config)
    answer = generate_answer(system_prompt, user_message, model_name, config)

    sources = sorted({c["page_title"] for c in chunks})
    if sources:
        answer += "\n\n**Sources :** " + ", ".join(sources)
    return answer


demo = gr.ChatInterface(
    respond,
    title="OneNoteRAG — Computer Science",
    description="Pose une question sur tes notes OneNote (thème Computer Science).",
    examples=[
        "Qu'est-ce qu'Ansible ?",
        "Quelle est la différence entre un processus et un thread ?",
    ],
)

if __name__ == "__main__":
    demo.launch()

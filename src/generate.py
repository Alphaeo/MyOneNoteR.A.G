"""
Étape 6 — Génération de la réponse avec Qwen2.5-1.5B local via Ollama
(GGUF quantifié, bien plus rapide sur CPU que les poids transformers bruts —
voir ARCHITECTURE.md), à partir des chunks rerankés par retrieve_rerank.py.

CLI simple : boucle de questions/réponses dans le terminal.

Prérequis : `ollama pull qwen2.5:1.5b` (ou le modèle configuré dans
config.yaml) et le serveur Ollama démarré (il tourne en arrière-plan après
installation sur Windows).
"""
import ollama
import yaml
from retrieve_rerank import retrieve
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

# TODO 1 — Construire le prompt système :
#   - Instructions : répondre UNIQUEMENT à partir du contexte fourni
#   - Demander de citer la page source pour chaque affirmation
#     (ex: "[Source: <page_title>]")
#   - Demander de répondre "Je ne sais pas d'après tes notes" si le contexte
#     ne permet pas de répondre (éviter les hallucinations)


def build_system_prompt() -> str:
    return (
        "Tu es un assistant qui répond à des questions à partir des notes "
        "personnelles OneNote de l'utilisateur (thème : Computer Science).\n\n"
        "Règles strictes :\n"
        "- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous, jamais "
        "à partir de tes connaissances générales.\n"
        "- Cite la page source de chaque affirmation avec la notation "
        "[Source: <titre de la page>].\n"
        "- Si le contexte ne permet pas de répondre à la question, réponds "
        "exactement : \"Je ne sais pas d'après tes notes.\" Ne devine jamais."
    )



# TODO 2 — Construire le message utilisateur en injectant les chunks
#   récupérés (texte + métadonnées de source), tronqué pour respecter la
#   fenêtre de contexte du modèle Qwen local utilisé.


def build_user_message(question: str, chunks: list[dict], config : dict) -> str:
    max_chunks = config["generation"]["max_context_chunks"]
    context_blocks = []
    for chunk in chunks[:max_chunks]:
        context_blocks.append(f"[Source: {chunk['page_title']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"Contexte :\n{context}\n\n---\n\nQuestion : {question}"


# Ollama gère lui-même le chargement/cache du modèle côté serveur — pas de
# vrai "chargement" côté client ici, juste le nom du modèle à réutiliser
# partout (pattern conservé pour cohérence avec les autres load_* du projet).


def load_generation_model(config: dict) -> str:
    return config["generation"]["model_name"]


def generate_answer(system_prompt: str, user_message: str, model_name: str, config: dict) -> str:
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        options={
            "temperature": config["generation"]["temperature"],
            "num_predict": config["generation"]["max_new_tokens"],
        },
    )
    return response["message"]["content"]


# TODO 4 — Boucle CLI :
#   1. Charger UNE FOIS (avant la boucle) : config, les 3 modèles de
#      retrieve_rerank (dense/sparse/cross_encoder) et load_generation_model()
#   2. Boucle : input(question) -> retrieve_rerank.retrieve(question, config,
#      dense_model, sparse_model, cross_encoder) -> build prompts ->
#      generate_answer() -> afficher la réponse + les sources utilisées
def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main() -> None:
    config = load_config()

    print("Chargement des modèles (peut prendre un moment)...")
    dense_model = SentenceTransformer(config["embedding"]["dense"]["model_name"])
    sparse_model = SparseTextEmbedding(model_name=config["embedding"]["sparse"]["model_name"])
    cross_encoder = CrossEncoder(config["reranker"]["model_name"], model_kwargs={"dtype": "bfloat16"})
    model_name = load_generation_model(config)
    system_prompt = build_system_prompt()

    print("Prêt. Pose ta question (Ctrl+C pour quitter).")
    while True:
        question = input("\n> ")
        chunks = retrieve(question, config, dense_model, sparse_model, cross_encoder)
        user_message = build_user_message(question, chunks, config)
        answer = generate_answer(system_prompt, user_message, model_name, config)
        print(f"\n{answer}")
        sources = sorted({c["page_title"] for c in chunks})
        print(f"\nSources : {', '.join(sources)}")


if __name__ == "__main__":
    main()

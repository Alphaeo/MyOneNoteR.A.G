"""
Étape 6 — Génération de la réponse avec Qwen2.5-1.5B-Instruct local (via
transformers, poids déjà en cache HF — pas d'Ollama), à partir des chunks
rerankés par retrieve_rerank.py.

CLI simple : boucle de questions/réponses dans le terminal.

Comme pour les modèles d'embedding/reranking, le modèle de génération doit
être chargé UNE SEULE FOIS (dans main(), avant la boucle de chat) et passé
en paramètre aux fonctions qui l'utilisent — pas rechargé à chaque question.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
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


# TODO 3a — Charger le tokenizer + le modèle UNE FOIS (appelé dans main(),
#   pas dans la boucle de chat) :
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   tokenizer = AutoTokenizer.from_pretrained(config["generation"]["model_name"])
#   model = AutoModelForCausalLM.from_pretrained(config["generation"]["model_name"])
#   Retourner (tokenizer, model).


def load_generation_model(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(config["generation"]["model_name"])
    # bfloat16 plutôt que float32 par défaut : ~moitié moins de RAM, important
    # sur une machine avec peu de RAM libre (voir contrainte notée ailleurs).
    model = AutoModelForCausalLM.from_pretrained(config["generation"]["model_name"], dtype="bfloat16")
    return tokenizer, model


# TODO 3b — Générer la réponse avec le modèle déjà chargé.
#   Qwen2.5-Instruct a un chat template intégré dans son tokenizer :
#   messages = [{"role": "system", "content": system_prompt},
#               {"role": "user", "content": user_message}]
#   input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
#   output_ids = model.generate(input_ids, max_new_tokens=config["generation"]["max_new_tokens"],
#                                temperature=config["generation"]["temperature"], do_sample=True)
#   Ne décoder QUE les tokens générés (pas le prompt réinjecté) :
#   response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)


def generate_answer(system_prompt: str, user_message: str, tokenizer, model, config: dict) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}]
    # apply_chat_template(..., return_tensors="pt") retourne un BatchEncoding
    # (dict-like), pas un tensor brut, sur les versions récentes de
    # transformers — on le déballe avec **inputs plutôt que de le passer
    # directement en premier argument positionnel.
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    output_ids = model.generate(**inputs, max_new_tokens=config["generation"]["max_new_tokens"],
                                 temperature=config["generation"]["temperature"], do_sample=True)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response


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
    tokenizer, model = load_generation_model(config)
    system_prompt = build_system_prompt()

    print("Prêt. Pose ta question (Ctrl+C pour quitter).")
    while True:
        question = input("\n> ")
        chunks = retrieve(question, config, dense_model, sparse_model, cross_encoder)
        user_message = build_user_message(question, chunks, config)
        answer = generate_answer(system_prompt, user_message, tokenizer, model, config)
        print(f"\n{answer}")
        sources = sorted({c["page_title"] for c in chunks})
        print(f"\nSources : {', '.join(sources)}")


if __name__ == "__main__":
    main()

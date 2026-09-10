"""
Étape 7 — Évaluation du pipeline RAG avec RAGAS.

Entrée  : eval/qa_dataset.jsonl (questions + réponses de référence, ÉCRITES
          À LA MAIN par toi — voir eval/qa_dataset.example.jsonl pour le format)
Sortie  : eval/results/ (scores par métrique, par question et en moyenne)

RAGAS par défaut appelle l'API OpenAI pour son LLM juge et ses embeddings.
Ici on le configure explicitement pour utiliser ton setup local : le juge
LLM passe par Ollama (même modèle Qwen2.5 que generate.py, bien plus rapide
sur CPU que les poids transformers bruts), et l'embedding juge (nécessaire
pour answer_relevancy) reste sur Qwen3-Embedding via transformers — sinon
ça échouera sans clé API.
"""
import json
import os
import datetime
import yaml
from retrieve_rerank import retrieve
from generate import build_system_prompt, build_user_message, generate_answer, load_generation_model
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# TODO 1 — Charger config.yaml (section "evaluation") et eval/qa_dataset.jsonl
#   Chaque ligne du dataset : {"question": ..., "ground_truth": ...}


def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# TODO 2 — Pour chaque question du dataset, faire tourner le pipeline complet :
#   - retrieve_rerank.retrieve(question, config, dense_model, sparse_model, cross_encoder) -> chunks
#   - contexts = [chunk["text"] for chunk in chunks]
#   - generate.build_system_prompt() / build_user_message() / generate_answer() -> answer
#   Collecter {"question", "answer", "contexts", "ground_truth"} pour
#   chaque ligne.
#
#   Les modèles (dense_model, sparse_model, cross_encoder) doivent être
#   chargés UNE FOIS par main() et passés ici en paramètres — pas rechargés
#   à chaque question du dataset. model_name ne nécessite pas de chargement
#   (Ollama gère ça côté serveur).


def run_pipeline_on_dataset(qa_pairs: list[dict], config: dict, dense_model, sparse_model, cross_encoder, model_name: str) -> list[dict]:
    system_prompt = build_system_prompt()
    results = []
    for qa in qa_pairs:
        question = qa["question"]
        chunks = retrieve(question, config, dense_model, sparse_model, cross_encoder)
        user_message = build_user_message(question, chunks, config)
        answer = generate_answer(system_prompt, user_message, model_name, config)
        results.append({
            "question": question,
            "answer": answer,
            "contexts": [c["text"] for c in chunks],
            "ground_truth": qa["ground_truth"],
        })
    return results


# TODO 3 — Construire un datasets.Dataset au format attendu par RAGAS.


def build_ragas_dataset(results: list[dict]):
    return Dataset.from_list([
        {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"], "ground_truth": r["ground_truth"]}
        for r in results
    ])


# Le juge LLM passe par Ollama (même modèle que generate.py, pas de
# rechargement/duplication puisque Ollama sert déjà le modèle en arrière-plan
# indépendamment du process Python). L'embedding juge (nécessaire pour
# answer_relevancy) reste chargé séparément via transformers.


def build_ragas_judges(config: dict, model_name: str):
    judge_llm = LangchainLLMWrapper(ChatOllama(
        model=model_name,
        base_url=config["generation"]["ollama_base_url"],
        temperature=config["generation"]["temperature"],
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config["evaluation"]["judge_embedding_model"])
    )
    return judge_llm, judge_embeddings


# TODO 5 — Lancer l'évaluation :
#   result = evaluate(dataset, metrics=[...], llm=judge_llm, embeddings=judge_embeddings)
#   result.to_pandas() -> DataFrame (PAS de to_csv()/to_json() directement
#   sur l'objet EvaluationResult, vérifié empiriquement — il faut passer par
#   to_pandas() d'abord).


def run_evaluation(dataset, judge_llm, judge_embeddings, config: dict) -> None:
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)

    results_dir = config["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = result.to_pandas()
    df.to_csv(f"{results_dir}/evaluation_{timestamp}.csv", index=False)

    print(df)
    print("\nMoyennes :")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    qa_pairs = load_eval_dataset(config["evaluation"]["dataset_path"])

    print("Chargement des modèles...")
    dense_model = SentenceTransformer(config["embedding"]["dense"]["model_name"])
    sparse_model = SparseTextEmbedding(model_name=config["embedding"]["sparse"]["model_name"])
    cross_encoder = CrossEncoder(config["reranker"]["model_name"], model_kwargs={"dtype": "bfloat16"})
    model_name = load_generation_model(config)

    print(f"Exécution du pipeline sur {len(qa_pairs)} questions...")
    results = run_pipeline_on_dataset(qa_pairs, config, dense_model, sparse_model, cross_encoder, model_name)
    dataset = build_ragas_dataset(results)

    judge_llm, judge_embeddings = build_ragas_judges(config, model_name)

    print("Évaluation RAGAS...")
    run_evaluation(dataset, judge_llm, judge_embeddings, config)


if __name__ == "__main__":
    main()

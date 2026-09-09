"""
Étape 7 — Évaluation du pipeline RAG avec RAGAS.

Entrée  : eval/qa_dataset.jsonl (questions + réponses de référence, ÉCRITES
          À LA MAIN par toi — voir eval/qa_dataset.example.jsonl pour le format)
Sortie  : eval/results/ (scores par métrique, par question et en moyenne)

RAGAS par défaut appelle l'API OpenAI pour son LLM juge et ses embeddings.
Ici on le configure explicitement pour utiliser ton setup local via
transformers (Qwen2.5-1.5B-Instruct, le même modèle que pour generate.py —
réutilisé tel quel, pas rechargé) plutôt qu'Ollama (jamais installé sur
cette machine) — sinon ça échouera sans clé API.
"""
import json
import os
import datetime
import yaml
from retrieve_rerank import retrieve
from generate import build_system_prompt, build_user_message, generate_answer, load_generation_model
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from transformers import pipeline as hf_pipeline
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
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
#   Les modèles (dense_model, sparse_model, cross_encoder, tokenizer, model)
#   doivent être chargés UNE FOIS par main() et passés ici en paramètres —
#   pas rechargés à chaque question du dataset.


def run_pipeline_on_dataset(qa_pairs: list[dict], config: dict, dense_model, sparse_model, cross_encoder, tokenizer, model) -> list[dict]:
    system_prompt = build_system_prompt()
    results = []
    for qa in qa_pairs:
        question = qa["question"]
        chunks = retrieve(question, config, dense_model, sparse_model, cross_encoder)
        user_message = build_user_message(question, chunks, config)
        answer = generate_answer(system_prompt, user_message, tokenizer, model, config)
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


# TODO 4 — Configurer le LLM juge et les embeddings pour pointer vers ton
#   setup local (transformers) plutôt qu'OpenAI.
#   Le juge réutilise le tokenizer/model déjà chargés pour generate_answer()
#   (même modèle Qwen2.5) — pas de rechargement, RAM limitée sur cette
#   machine. Seul l'embedding juge (nécessaire pour answer_relevancy)
#   nécessite un chargement séparé.


def build_ragas_judges(config: dict, tokenizer, model):
    text_gen_pipeline = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config["generation"]["max_new_tokens"],
    )
    judge_llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=text_gen_pipeline))
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
    tokenizer, model = load_generation_model(config)

    print(f"Exécution du pipeline sur {len(qa_pairs)} questions...")
    results = run_pipeline_on_dataset(qa_pairs, config, dense_model, sparse_model, cross_encoder, tokenizer, model)
    dataset = build_ragas_dataset(results)

    judge_llm, judge_embeddings = build_ragas_judges(config, tokenizer, model)

    print("Évaluation RAGAS...")
    run_evaluation(dataset, judge_llm, judge_embeddings, config)


if __name__ == "__main__":
    main()

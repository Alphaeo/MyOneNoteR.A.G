"""
Étape 3 — Découpage des pages Markdown en chunks indexables.

INCRÉMENTAL : un fichier de chunks PAR PAGE plutôt qu'un seul gros fichier,
pour ne rechunker que les pages qui ont changé (voir data/raw/last_sync.json
produit par extract_onenote.py, et ARCHITECTURE.md section "Synchronisation
incrémentale").

Entrée  : data/processed/<page_id>.md
Sortie  : data/processed/chunks/<page_id>.jsonl
          une ligne JSON par chunk :
          {"chunk_id", "page_id", "page_title", "source_url", "notebook",
           "section", "text", "chunk_index"}

Stratégie recommandée (voir ARCHITECTURE.md) : découpe par heading Markdown,
avec les blocs de code gardés atomiques, et overlap seulement si une section
dépasse max_tokens_per_chunk.
"""
import yaml
import json
import glob
import os

# TODO 1 — Charger config.yaml (section "chunking") pour les paramètres
#   (strategy, max_tokens_per_chunk, overlap_ratio, keep_code_blocks_atomic)


def load_chunking_config() -> dict:
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("chunking", {})


# TODO 2 — Parser un document Markdown en une liste de "sections" délimitées
#   par les headings (#, ##, ###...). Chaque section garde son titre de heading
#   pour le contexte (utile à réinjecter dans le chunk final, ex: "## Section >
#   texte").


def split_by_heading(markdown_text: str) -> list[dict]:
    sections = []
    current_section = {"title": "", "content": ""}
    in_code_block = False
    for line in markdown_text.splitlines():
        if line.strip().startswith("```"):
            # Bascule dedans/dehors d'un bloc de code — un "#" à l'intérieur
            # (ex: commentaire Python) ne doit jamais être pris pour un heading.
            in_code_block = not in_code_block
            current_section["content"] += line + "\n"
            continue
        if line.startswith("#") and not in_code_block:
            # C'est un heading, créer une nouvelle section
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"title": line, "content": ""}
        else:
            # Ajouter la ligne à la section courante
            current_section["content"] += line + "\n"
    if current_section["content"]:
        sections.append(current_section)
    return sections


# TODO 3 — Repérer les blocs de code (```...```) et les traiter comme unités
#   atomiques : ne jamais les couper au milieu, même si ça dépasse
#   max_tokens_per_chunk.
#
#   Retourne une liste de SEGMENTS alternant texte et code (pas juste les
#   blocs de code seuls) — c'est ce qui permet à chunk_section_content()
#   ci-dessous de découper le texte librement tout en gardant chaque bloc de
#   code entier.


def extract_code_blocks(section_text: str) -> list[dict]:
    segments = []
    current_text = ""
    current_code = ""
    in_code_block = False
    for line in section_text.splitlines():
        if line.strip().startswith("```"):
            if not in_code_block:
                # Ouverture d'un bloc : on ferme le segment de texte en cours
                if current_text:
                    segments.append({"type": "text", "content": current_text})
                    current_text = ""
                current_code = line + "\n"
            else:
                # Fermeture du bloc : on le referme et l'ajoute comme segment atomique
                current_code += line + "\n"
                segments.append({"type": "code", "content": current_code})
                current_code = ""
            in_code_block = not in_code_block
            continue
        if in_code_block:
            current_code += line + "\n"
        else:
            current_text += line + "\n"
    if current_text:
        segments.append({"type": "text", "content": current_text})
    if current_code:
        # Bloc de code jamais refermé (fence manquante) — on le garde quand même
        segments.append({"type": "code", "content": current_code})
    return segments


# TODO 4 — Si une section (hors blocs de code) dépasse max_tokens_per_chunk,
#   la sous-découper avec un overlap de overlap_ratio.
#   Utiliser un tokenizer cohérent avec ton modèle d'embedding pour compter
#   les tokens (ex: tokenizer HF du modèle Qwen3-Embedding choisi) — ici on
#   utilise .split() sur les espaces comme approximation simple pour
#   démarrer, à améliorer plus tard si besoin.


def split_long_section(section_text: str, max_tokens: int, overlap_ratio: float) -> list[str]:
    tokens = section_text.split()
    if not tokens:
        return []
    step = max(1, max_tokens - int(max_tokens * overlap_ratio))
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk_tokens))
        if i + max_tokens >= len(tokens):
            break
        i += step
    return chunks


# Combine extract_code_blocks() + split_long_section() : découpe le texte
# normal par tokens (avec overlap), mais garde chaque bloc de code comme un
# chunk unique et entier, quelle que soit sa taille.


def chunk_section_content(section_text: str, config: dict) -> list[str]:
    max_tokens = config["max_tokens_per_chunk"]
    overlap_ratio = config["overlap_ratio"]

    if not config.get("keep_code_blocks_atomic", True):
        return split_long_section(section_text, max_tokens, overlap_ratio)

    chunks = []
    for segment in extract_code_blocks(section_text):
        if segment["type"] == "code":
            chunks.append(segment["content"])
        else:
            chunks.extend(split_long_section(segment["content"], max_tokens, overlap_ratio))
    return chunks


# TODO 5 — Assembler les chunks d'UNE page avec leurs métadonnées de
#   provenance et écrire data/processed/chunks/<page_id>.jsonl (un chunk par
#   ligne JSON). Une fonction séparée pour une seule page, réutilisable en
#   mode incrémental comme en mode complet.


def chunk_page(page_id: str, markdown_text: str, metadata: dict, config: dict) -> list[dict]:
    sections = split_by_heading(markdown_text)
    chunks = []
    chunk_index = 0  # compteur global à la page, PAS remis à zéro par section
    for section in sections:
        for text in chunk_section_content(section["content"], config):
            chunks.append({
                "chunk_id": f"{page_id}_{chunk_index}",
                "page_id": page_id,
                "page_title": metadata["title"],
                "source_url": metadata["source_url"],
                "notebook": metadata["notebook"],
                "section": section["title"],
                "text": text,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
    return chunks


def write_page_chunks(page_id: str, chunks: list[dict]) -> None:
    os.makedirs("data/processed/chunks", exist_ok=True)
    with open(f"data/processed/chunks/{page_id}.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# TODO 6 — Orchestration incrémentale :
#   1. Lire data/raw/last_sync.json (produit par extract_onenote.py)
#   2. Pour chaque page_id dans added + modified :
#      - lire data/processed/<page_id>.md (déjà généré par parse_clean.py)
#      - chunk_page() puis write_page_chunks()
#   3. Pour chaque page_id dans deleted :
#      - supprimer data/processed/chunks/<page_id>.jsonl s'il existe
#   4. Si last_sync.json n'existe pas (premier run, ou tu veux tout
#      retraiter) : boucler sur TOUS les fichiers data/processed/*.md à la
#      place — prévoir un flag --full pour ce cas, comme dans
#      extract_onenote.py.


def load_last_sync(path: str = "data/raw/last_sync.json") -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(page_id: str) -> dict:
    with open(f"data/raw/{page_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config = load_chunking_config()
    last_sync = load_last_sync()
    full_reprocess = last_sync is None
    if full_reprocess:
        last_sync = {}

    for page_id in last_sync.get("added", []) + last_sync.get("modified", []):
        with open(f"data/processed/{page_id}.md", "r", encoding="utf-8") as f:
            markdown_text = f.read()
        metadata = load_metadata(page_id)
        chunks = chunk_page(page_id, markdown_text, metadata, config)
        write_page_chunks(page_id, chunks)

    for page_id in last_sync.get("deleted", []):
        try:
            os.remove(f"data/processed/chunks/{page_id}.jsonl")
        except FileNotFoundError:
            pass

    if full_reprocess:
        for md_file in glob.glob("data/processed/*.md"):
            page_id = os.path.splitext(os.path.basename(md_file))[0]
            with open(md_file, "r", encoding="utf-8") as f:
                markdown_text = f.read()
            metadata = load_metadata(page_id)
            chunks = chunk_page(page_id, markdown_text, metadata, config)
            write_page_chunks(page_id, chunks)

    print("Chunking terminé.")


if __name__ == "__main__":
    main()

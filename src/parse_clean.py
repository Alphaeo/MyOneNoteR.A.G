"""
Étape 2 — Nettoyage du HTML OneNote -> Markdown propre.

INCRÉMENTAL : ne traiter que ce qui a changé, via data/raw/last_sync.json
produit par extract_onenote.py (voir ARCHITECTURE.md section
"Synchronisation incrémentale").

Entrée  : data/raw/*.html + *.json
Sortie  : data/processed/*.md (avec front-matter YAML pour les métadonnées)

Le HTML OneNote a des particularités à gérer :
    - <div style="position:absolute; ..."> pour le placement des blocs
    - des <span style="..."> de mise en forme sans valeur sémantique
    - des blocs de code parfois en <pre>, parfois en texte simple avec des
      espaces insécables — à repérer et vérifier au cas par cas sur tes vraies pages
    - des images en <img data-src-type="image/png" src="...">
"""
import json
import os
from bs4 import BeautifulSoup
from markdownify import markdownify as md_convert

# Caractère que OneNote utilise pour marquer un saut de ligne À L'INTÉRIEUR
# d'un bloc de code (chaque span Consolas qui ne contient QUE ce caractère
# représente une fin de ligne). Confirmé empiriquement sur data/raw/*.html
# (voir la page "Ansible") — U+FFFC = OBJECT REPLACEMENT CHARACTER.
CODE_LINE_BREAK_MARKER = "￼"

# TODO 1 — Charger le HTML brut + les métadonnées JSON associées


def load_raw_page(page_id: str) -> tuple[str, dict]:
    with open(f"data/raw/{page_id}.html", "r", encoding="utf-8") as f:
        html = f.read()
    with open(f"data/raw/{page_id}.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return html, metadata


# TODO 2 — Parser avec BeautifulSoup et remonter la structure sémantique :
#   - <span style="font-weight:bold/font-style:italic"> -> <strong>/<em>
#     (OneNote n'utilise JAMAIS <strong>/<em> directement, donc sans cette
#     étape markdownify ne mettra rien en gras/italique)
#   - <p> dont TOUS les spans enfants sont en police Consolas -> bloc de code
#     (reconstruction des lignes via CODE_LINE_BREAK_MARKER)
#   - span Consolas isolé au milieu d'un paragraphe normal -> code inline
#   - seulement APRÈS ces conversions sémantiques : supprimer les attributs
#     style restants (les garder plus tôt les aurait empêchées, puisque
#     c'est justement le style qui indique bold/italic/code)


def clean_html(raw_html: str) -> BeautifulSoup:
    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. Blocs de code : <p> où 100% des spans enfants sont en Consolas
    for p in soup.find_all("p"):
        spans = p.find_all("span", recursive=False)
        if not spans:
            continue
        if not all("consolas" in (s.get("style") or "").lower() for s in spans):
            continue

        lines = []
        current_line = ""
        for span in spans:
            text = span.get_text()
            if text == CODE_LINE_BREAK_MARKER:
                lines.append(current_line)
                current_line = ""
            else:
                current_line += text
        if current_line:
            lines.append(current_line)

        pre = soup.new_tag("pre")
        code = soup.new_tag("code")
        code.string = "\n".join(lines)
        pre.append(code)
        p.replace_with(pre)

    # 2. Code inline : spans Consolas isolés restants (dans du texte normal)
    for span in soup.find_all("span", style=True):
        if "consolas" in span["style"].lower():
            code_tag = soup.new_tag("code")
            code_tag.string = span.get_text()
            span.replace_with(code_tag)

    # 3. Gras / italique (gère aussi le cas gras+italique combinés)
    for span in soup.find_all("span", style=True):
        style = span["style"].lower()
        is_bold = "font-weight:bold" in style
        is_italic = "font-style:italic" in style
        if not is_bold and not is_italic:
            continue

        text = span.get_text()
        if is_bold and is_italic:
            new_tag = soup.new_tag("strong")
            em = soup.new_tag("em")
            em.string = text
            new_tag.append(em)
        elif is_bold:
            new_tag = soup.new_tag("strong")
            new_tag.string = text
        else:
            new_tag = soup.new_tag("em")
            new_tag.string = text
        span.replace_with(new_tag)

    # 4. Supprimer les attributs style restants (plus besoin, tout ce qui
    #    devait être converti sémantiquement l'a déjà été)
    for tag in soup.find_all(True):
        if "style" in tag.attrs:
            del tag.attrs["style"]

    # 5. Supprimer les balises vides résiduelles (mais pas <br>/<img>, qui
    #    sont "vides" par nature et portent quand même une information)
    for tag in soup.find_all(True):
        if not tag.contents and not tag.string and tag.name not in ("br", "img"):
            tag.decompose()

    return soup


# TODO 3 — Convertir en Markdown via markdownify.
#   heading_style="ATX" force les titres "# / ##" plutôt que le style
#   "underline" (=== / ---), plus fiable pour du Markdown généré.
#   IMPORTANT : teste bien le rendu des tableaux sur une page qui en contient
#   (ex: la page "Ansible") — certaines versions de markdownify gèrent mal
#   les <table> par défaut.


def to_markdown(cleaned_soup: BeautifulSoup) -> str:
    # Ne convertir que le <body> : passer le document entier faisait fuiter
    # le contenu de <title> (dans <head>) comme ligne de texte parasite en
    # tête de chaque page — markdownify ne sait pas l'ignorer tout seul.
    body = cleaned_soup.body or cleaned_soup
    markdown = md_convert(str(body), heading_style="ATX")
    # markdownify laisse souvent 3+ lignes vides d'affilée (héritage du
    # positionnement absolu de OneNote) ; on les compresse à 2 max pour la
    # lisibilité du fichier .md final.
    while "\n\n\n" in markdown:
        markdown = markdown.replace("\n\n\n", "\n\n")
    return markdown.strip() + "\n"


# TODO 4 — Images : décision prise = option simple, on les ignore.
#   markdownify convertit un <img> en "![](url)" — comme on n'a pas
#   téléchargé les images (extract_onenote.py ne récupère que le HTML),
#   ces liens seraient de toute façon cassés. Si tu veux les gérer plus tard
#   (OCR ou description via modèle vision), c'est ici qu'il faudra intervenir,
#   AVANT l'appel à md_convert dans to_markdown (remplacer <img> par un texte
#   descriptif dans clean_html()).


# TODO 5 — Écrire le fichier .md avec front-matter YAML (fait ci-dessous)


def write_processed_page(page_id: str, metadata: dict, markdown: str) -> None:
    front_matter = (
        "---\n"
        f"title: \"{metadata['title']}\"\n"
        f"source_url: \"{metadata['source_url']}\"\n"
        f"notebook: \"{metadata['notebook']}\"\n"
        f"section: \"{metadata['section']}\"\n"
        f"last_modified: \"{metadata['lastModifiedDateTime']}\"\n"
        "---\n\n"
    )
    os.makedirs("data/processed", exist_ok=True)
    with open(f"data/processed/{page_id}.md", "w", encoding="utf-8") as f:
        f.write(front_matter + markdown)


def main() -> None:
    last_sync_path = "data/raw/last_sync.json"
    if os.path.exists(last_sync_path):
        with open(last_sync_path, "r", encoding="utf-8") as f:
            last_sync = json.load(f)
        to_process = last_sync["added"] + last_sync["modified"]
        to_delete = last_sync["deleted"]
    else:
        # Pas de last_sync.json (premier run / mode --full) : tout traiter
        to_process = [
            fname[: -len(".html")]
            for fname in os.listdir("data/raw")
            if fname.endswith(".html")
        ]
        to_delete = []

    for page_id in to_process:
        html, metadata = load_raw_page(page_id)
        cleaned = clean_html(html)
        markdown = to_markdown(cleaned)
        write_processed_page(page_id, metadata, markdown)

    for page_id in to_delete:
        md_path = f"data/processed/{page_id}.md"
        if os.path.exists(md_path):
            os.remove(md_path)

    print(f"{len(to_process)} pages parsées, {len(to_delete)} supprimées")


if __name__ == "__main__":
    main()

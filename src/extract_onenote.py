"""
Étape 1 — Extraction du notebook OneNote "Computer Science" via Microsoft Graph API.

INCRÉMENTAL : ce notebook change tous les jours, donc ce script ne doit
retélécharger que ce qui a changé depuis le dernier run, pas tout à chaque
fois. Voir ARCHITECTURE.md section "Synchronisation incrémentale" pour le
détail du mécanisme.

Sortie attendue :
    data/raw/<page_id>.html    — contenu HTML brut de chaque page
    data/raw/<page_id>.json    — métadonnées (title, id, notebook, section,
                                  lastModifiedDateTime, oneNoteWebUrl)
    data/raw/manifest.json     — {page_id: lastModifiedDateTime} de TOUT ce
                                  qui a déjà été extrait, pour savoir au
                                  prochain run ce qui a changé
    data/raw/last_sync.json    — delta du dernier run :
                                  {"added": [...], "modified": [...],
                                   "deleted": [...], "timestamp": ...}
                                  Consommé par parse_clean.py, chunk.py et
                                  embed_index.py pour ne retraiter QUE les
                                  pages concernées, pas tout le notebook.

Référence API : https://learn.microsoft.com/en-us/graph/api/resources/onenote-api-overview
"""
import os
import msal
from dotenv import load_dotenv
import requests
import time
import json
import datetime

load_dotenv()  # Charger les variables d'environnement depuis .env

AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
SCOPES = ["Notes.Read"]
TOKEN_CACHE_FILE = "token_cache.bin" 

# TODO 1 — Authentification MSAL (device code flow)
#   - Créer un PublicClientApplication avec AZURE_CLIENT_ID / AZURE_TENANT_ID (depuis .env)
#   - Utiliser acquire_token_by_device_flow() pour la première connexion
#   - Mettre en cache le token (msal.SerializableTokenCache) dans un fichier local
#     pour ne pas redemander une connexion à chaque exécution
#   - Scope : ["Notes.Read"]
#
# Doc MSAL device flow :
# https://learn.microsoft.com/en-us/entra/identity-platform/scenario-desktop-acquire-token-device-code-flow


def get_access_token() -> str:
    """Retourne un access token valide (depuis le cache ou via device code flow)."""
    cache = msal.SerializableTokenCache()
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, "r") as f:
            cache.deserialize(f.read())

    app = msal.PublicClientApplication(
        AZURE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
        token_cache=cache,
    )

    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Échec du démarrage du device flow : {flow}")
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)

    if cache.has_state_changed:
        with open(TOKEN_CACHE_FILE, "w") as f:
            f.write(cache.serialize())
    if "access_token" not in result:
        raise RuntimeError(f"Authentification échouée : {result.get('error_description')}")     
    return result["access_token"]



    


# TODO 2 — Lister les notebooks et trouver celui qui correspond à
#   ONENOTE_NOTEBOOK_NAME (depuis .env), insensible à la casse.
#   Endpoint : GET https://graph.microsoft.com/v1.0/me/onenote/notebooks


def find_notebook(access_token: str, notebook_name: str) -> dict:
    url = "https://graph.microsoft.com/v1.0/me/onenote/notebooks"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    notebooks = response.json().get("value", [])
    for notebook in notebooks:
        if notebook.get("displayName", "").lower() == notebook_name.lower():
            return notebook
    raise ValueError(f"Notebook '{notebook_name}' non trouvé.")


# TODO 3 — Lister toutes les sections du notebook (et les section groups si tu en as).
#   Endpoint : GET /me/onenote/notebooks/{id}/sections
#   Attention aux section groups imbriqués : /me/onenote/sectionGroups/{id}/sections


def list_sections(access_token: str, notebook_id: str) -> list[dict]:
    url = f"https://graph.microsoft.com/v1.0/me/onenote/notebooks/{notebook_id}/sections"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("value", [])


# TODO 4 — Lister toutes les pages d'une section, avec pagination.
#   Endpoint : GET /me/onenote/sections/{id}/pages
#   Gérer @odata.nextLink tant qu'il est présent dans la réponse.


def list_pages(access_token: str, section_id: str) -> list[dict]:
    url = f"https://graph.microsoft.com/v1.0/me/onenote/sections/{section_id}/pages"
    headers = {"Authorization": f"Bearer {access_token}"}
    pages = []
    while url:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        pages.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
    return pages


# TODO 5 — Récupérer le contenu HTML d'une page.
#   Endpoint : GET /me/onenote/pages/{id}/content
#   Gérer le rate limiting (HTTP 429 -> lire le header Retry-After et réessayer)


def get_page_content(access_token: str, page_id: str) -> str:
    url = f"https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}/content"
    headers = {"Authorization": f"Bearer {access_token}"}
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        return response.text


# TODO 6a — Charger/sauvegarder le manifest (data/raw/manifest.json).
#   Format : {page_id: lastModifiedDateTime_string}
#   Si le fichier n'existe pas encore (tout premier run), retourner {}.


def load_manifest(path: str = "data/raw/manifest.json") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: str = "data/raw/manifest.json") -> None:
    with open(path, "w") as f:
        return json.dump(manifest, f)


# TODO 6b — Comparer les pages actuelles (retournées par l'API, via
#   list_pages sur toutes les sections) au manifest existant, pour décider
#   quoi retélécharger.
#
#   added    = pages dont l'id n'est pas dans le manifest
#   modified = pages dont l'id est dans le manifest MAIS dont
#              page["lastModifiedDateTime"] != manifest[page_id]
#   deleted  = ids présents dans le manifest mais absents des pages actuelles
#              (= supprimées côté OneNote depuis le dernier run)
#
#   Retourne un tuple (added, modified, deleted) — added et modified sont des
#   listes de dicts "page" (objets Graph API complets, tu auras besoin du
#   contenu), deleted est juste une liste de page_id (plus rien à
#   télécharger, juste à nettoyer).


def compute_changes(current_pages: list[dict], manifest: dict) -> tuple[list[dict], list[dict], list[str]]:
    added = [page for page in current_pages if page["id"] not in manifest]
    modified = [page for page in current_pages if page["id"] in manifest and page["lastModifiedDateTime"] != manifest[page["id"]]]
    deleted = [page_id for page_id in manifest if page_id not in [page["id"] for page in current_pages]]
    return added, modified, deleted


# TODO 6c — Orchestration complète :
#   1. token, notebook
#   2. Parcourir toutes les sections -> toutes les pages (list_sections + list_pages)
#      Construis une liste plate de tous les objets "page", en gardant une
#      référence au nom de la section parente pour chacun (tu en auras besoin
#      dans les métadonnées).
#   3. charger le manifest existant, calculer (added, modified, deleted) via
#      compute_changes()
#   4. Pour chaque page dans added + modified :
#      - get_page_content() -> écrire data/raw/{page_id}.html
#      - écrire data/raw/{page_id}.json (title, id, notebook, section,
#        lastModifiedDateTime, source_url = page["links"]["oneNoteWebUrl"]["href"])
#      - mettre à jour le manifest en mémoire (page_id -> lastModifiedDateTime)
#   5. Pour chaque page_id dans deleted :
#      - supprimer data/raw/{page_id}.html et .json s'ils existent
#      - retirer l'entrée du manifest en mémoire
#   6. save_manifest(manifest)
#   7. Écrire data/raw/last_sync.json avec :
#      {"added": [ids], "modified": [ids], "deleted": [ids], "timestamp": ...}
#      (juste les ids, pas les objets complets — les étapes suivantes du
#      pipeline liront directement les .html/.json/.md déjà sur disque)
#   8. Affiche un résumé (ex: "12 ajoutées, 3 modifiées, 1 supprimée, 204 inchangées")
#
#   Si tu veux forcer une extraction complète (ex: premier run, ou pour
#   debug), tu peux prévoir un flag --full qui ignore le manifest et traite
#   tout comme "added".


def main() -> None:
    token = get_access_token()
    notebook = find_notebook(token, os.getenv('ONENOTE_NOTEBOOK_NAME'))
    sections = list_sections(token, notebook['id'])
    print(f"{len(sections)} sections trouvées")
    all_pages = []
    for section in sections:
        pages = list_pages(token, section["id"])
        for page in pages:
            page["_section_name"] = section["displayName"]
            all_pages.append(page)
        print(f"  {len(pages)} pages dans la section '{section['displayName']}'")

    manifest = load_manifest()
    added, modified, deleted = compute_changes(all_pages, manifest)

    # Télécharger les pages ajoutées et modifiées
    for page in added + modified:
        html_content = get_page_content(token, page["id"])
        with open(f"data/raw/{page['id']}.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        metadata = {
            "title": page["title"],
            "id": page["id"],
            "notebook": notebook["displayName"],
            "section": page["_section_name"],
            "lastModifiedDateTime": page["lastModifiedDateTime"],
            "source_url": page["links"]["oneNoteWebUrl"]["href"]
        }
        with open(f"data/raw/{page['id']}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        manifest[page["id"]] = page["lastModifiedDateTime"]
    
    # Supprimer les pages supprimées
    for page_id in deleted:
        html_path = f"data/raw/{page_id}.html"
        json_path = f"data/raw/{page_id}.json"
        if os.path.exists(html_path):
            os.remove(html_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        manifest.pop(page_id, None)

    print(f"{len(added)} pages ajoutées")
    print(f"{len(modified)} pages modifiées")
    print(f"{len(deleted)} pages supprimées")

    last_sync = {
        "added": [page["id"] for page in added],
        "modified": [page["id"] for page in modified],
        "deleted": deleted,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open("data/raw/last_sync.json", "w", encoding="utf-8") as f:
        json.dump(last_sync, f, ensure_ascii=False, indent=2)

    save_manifest(manifest)


if __name__ == "__main__":
    main()


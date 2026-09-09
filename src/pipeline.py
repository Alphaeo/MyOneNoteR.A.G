"""
Orchestrateur optionnel — à écrire une fois que chaque étape fonctionne
individuellement en isolation (recommandé : teste chaque script séparément
d'abord, avec un petit sous-ensemble de pages, avant de tout enchaîner ici).

Enchaînement prévu :
    extract_onenote.main()
    parse_clean.main()
    chunk.main()
    embed_index.main()
    (generate.py reste interactif, pas dans le pipeline batch)
"""

# TODO — une fois chaque module validé indépendamment, importer et enchaîner
# les .main() ci-dessus. Ajouter des logs/timers pour voir où le temps passe
# (l'extraction Graph API et l'embedding sont probablement les étapes les
# plus lentes).
import extract_onenote
import parse_clean
import chunk
import embed_index

def main() -> None:
    extract_onenote.main()
    parse_clean.main()
    chunk.main()
    embed_index.main()


if __name__ == "__main__":
    main()

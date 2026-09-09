# Architecture & décisions

## 1. Extraction — Microsoft Graph API

- Endpoint clé : `GET https://graph.microsoft.com/v1.0/me/onenote/notebooks`
  puis `.../sections` puis `.../pages`, ou directement filtrer par nom de
  notebook ("Computer Science") avec `$filter`.
- Le contenu d'une page s'obtient via `GET /me/onenote/pages/{id}/content`
  (retourne du HTML, pas du Markdown).
- Auth : MSAL Python, **device code flow** (pas besoin de secret client, pas de
  redirect URI web à gérer) — adapté à un usage perso en script.
- Scope : `Notes.Read` (lecture seule, suffisant).
- Pense à gérer :
  - la pagination (`@odata.nextLink`)
  - le rate limiting (Graph renvoie 429 avec `Retry-After`)
  - le cache du token MSAL (fichier local) pour ne pas re-login à chaque run
- Sauvegarde le HTML brut + les métadonnées (titre, id, section, notebook,
  lastModifiedDateTime, `links.oneNoteWebUrl` pour pouvoir citer/retrouver la
  page source) dans `data/raw/`.

**Alternative plus simple pour démarrer** : export manuel de quelques pages en
`.docx`/HTML depuis OneNote, si tu veux tester le reste du pipeline avant de
t'attaquer à l'auth Graph.

## 1bis. Synchronisation incrémentale

Contexte : le notebook change tous les jours. Re-télécharger, re-parser,
re-chunker et ré-embedder les ~200+ pages à chaque run serait lent et
inutile — on ne veut retraiter que ce qui a réellement changé.

**Mécanisme** :

1. **Manifest** (`data/raw/manifest.json`) : un simple `{page_id:
   lastModifiedDateTime}` de tout ce qui a déjà été extrait. Après chaque
   run réussi, il reflète l'état "connu" du notebook.
2. À chaque run, `extract_onenote.py` liste **toutes** les pages actuelles
   (léger — pas de téléchargement de contenu, juste les métadonnées) et
   compare à ce manifest :
   - **added** : `page_id` absent du manifest → nouvelle page
   - **modified** : `page_id` présent mais `lastModifiedDateTime` différent
     → page éditée depuis le dernier run
   - **deleted** : `page_id` présent dans le manifest mais absent de la
     liste actuelle → page supprimée côté OneNote
3. Seules les pages `added`/`modified` sont effectivement téléchargées
   (`get_page_content`). Les pages `deleted` sont juste nettoyées localement.
4. Le manifest est mis à jour et sauvegardé, et un **fichier delta**
   (`data/raw/last_sync.json`) est écrit avec juste les listes d'ids
   `{added, modified, deleted, timestamp}`.
5. **Chaque étape suivante du pipeline lit ce delta** et ne retraite que ce
   qu'il désigne :
   - `parse_clean.py` : reparse `added+modified`, supprime les `.md` des
     pages `deleted`
   - `chunk.py` : produit **un fichier de chunks par page**
     (`data/processed/chunks/<page_id>.jsonl`, pas un unique gros fichier)
     — ça permet de ne rechunker que les pages concernées, et de savoir
     précisément quel fichier supprimer pour une page effacée
   - `embed_index.py` : pour `modified`+`deleted`, supprime d'abord les
     points Qdrant de cette page via un filtre sur le payload `page_id`
     (`client.delete(..., points_selector=Filter(...))`), puis pour
     `added`+`modified`, embedde et upsert les nouveaux chunks

**Pourquoi ce découpage plutôt qu'un simple hash de fichier ou un check de
mtime** : l'API Graph donne `lastModifiedDateTime` gratuitement dans le même
appel qui liste les pages — pas besoin de télécharger le contenu pour savoir
si quelque chose a changé, ce qui rend le diff quasi instantané même sur un
gros notebook.

**Cas particulier — premier run** : le manifest n'existe pas encore, donc
tout est classé `added`. Prévoir aussi un flag `--full` sur chaque script
pour forcer un retraitement complet à la demande (ex: après avoir changé de
modèle d'embedding, ou pour débugger).

**Usage au quotidien** : relance simplement les 4 scripts dans l'ordre
(`extract_onenote.py`, `parse_clean.py`, `chunk.py`, `embed_index.py`)
chaque fois que tu veux resynchroniser — chacun ne fera le travail que sur
ce qui a changé depuis la dernière fois. Tu peux aussi automatiser ça
(tâche planifiée quotidienne) une fois que tu as confiance dans le
pipeline.

## 2. Parsing / nettoyage

Le HTML OneNote est verbeux (positionnement absolu en `style="position:absolute"`,
classes générées, images en `<img data-src-type="image/png" ...>`). Objectif :
en extraire du texte/Markdown propre.

- `BeautifulSoup` pour parser
- `markdownify` (ou conversion maison) pour HTML → Markdown
- Garder les blocs `<pre>`/`<code>` intacts (souvent tes snippets de code CS)
- Décider quoi faire des images : les ignorer, ou les décrire via un modèle
  vision si tu veux plus tard indexer des schémas/diagrammes
- Un fichier `.md` par page, avec un front-matter YAML (titre, source_url, date)

## 3. Chunking

Pour du contenu "Computer Science" (cours, définitions, code), le chunking par
**taille fixe en tokens** casse souvent les exemples de code ou les
définitions au milieu. Préférer :

- Découpe primaire par **heading** (`#`, `##` du Markdown généré)
- Si une section est trop longue, sous-découpe avec un overlap (~10-15%)
- Ne jamais couper à l'intérieur d'un bloc ``` code ``` — le garder comme
  chunk atomique, quitte à le sur-dimensionner
- Chaque chunk garde ses métadonnées de provenance (page, section, notebook,
  position) pour la citation finale

Sortie : `data/processed/chunks/<page_id>.jsonl` — **un fichier par page**
(plutôt qu'un seul gros fichier), une ligne JSON par chunk
(`{"id", "page_id", "text", "page_title", "source_url", "notebook", ...}`).
Le découpage par page est ce qui permet la synchronisation incrémentale
(section 1bis) : rechunker une page = réécrire un seul petit fichier.

## 4. Embeddings — dense (Qwen3-Embedding) + sparse (BM25)

### Dense

- Modèles dispo : `Qwen3-Embedding-0.6B` (rapide, CPU-friendly),
  `-4B`/`-8B` (meilleure qualité, besoin de GPU/RAM)
- Deux chemins d'implémentation :
  1. Si Ollama sert un modèle d'embedding compatible → `POST /api/embeddings`
  2. Sinon → `sentence-transformers` avec les poids HF directement
     (`SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")`)
- Qwen3-Embedding attend un **prompt d'instruction** différent pour les
  documents vs les requêtes (`"query: ..."` / `"document: ..."` ou instruction
  dédiée selon la doc du modèle) — à vérifier dans la carte HF du modèle,
  c'est important pour la qualité de la recherche.
- Normaliser les vecteurs (cosine similarity) avant indexation.

### Sparse

- Recherche dense et sparse sont complémentaires : le dense capture le sens
  ("qu'est-ce qu'un tri par tas"), le sparse capture les correspondances
  lexicales exactes (acronymes, noms de fonctions, mots-clés précis type
  `O(n log n)`, `TCP`, `mutex`) que les embeddings denses ratent parfois.
- Défaut recommandé : **BM25** via `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")`
  — pas de modèle de langage, pas de GPU, calcul quasi instantané, très
  robuste sur du contenu technique.
- Alternative plus coûteuse mais parfois plus précise : un modèle **SPLADE**
  (sparse appris) via fastembed également — mêmes interfaces, juste changer
  `model_name`. À essayer seulement si BM25 déçoit en pratique.
- Le vecteur sparse produit est une paire `(indices, values)` — un point par
  token présent dans le texte, pas une taille fixe comme le dense.

## 5. Base vectorielle — Qdrant (hybrid search : dense + sparse)

- Deux modes possibles :
  - **Embarqué (recommandé pour démarrer)** : `QdrantClient(path="./data/index")`
    — pas de Docker, pas de serveur à lancer, persiste sur disque comme
    Chroma. C'est le mode configuré par défaut dans `config.yaml`.
  - **Serveur** (Docker `qdrant/qdrant`) : `QdrantClient(host=..., port=6333)`
    — utile si tu veux plus tard une UI d'inspection (dashboard web intégré
    sur `:6333/dashboard`) ou des writes concurrents depuis plusieurs process
- Qdrant supporte nativement les **vecteurs nommés** : une collection peut
  stocker un vecteur dense ET un vecteur sparse par point, sous des noms
  différents (`"dense"`, `"sparse"`) :
  ```python
  client.create_collection(
      collection_name,
      vectors_config={"dense": VectorParams(size=<dim>, distance=Distance.COSINE)},
      sparse_vectors_config={"sparse": SparseVectorParams()},
  )
  ```
  La dimension `<dim>` doit correspondre exactement à la sortie du modèle
  d'embedding dense choisi (à vérifier sur sa carte HF).
- **Recherche hybride** via l'API `query_points` avec `prefetch` (une branche
  par type de vecteur) + `FusionQuery` :
  ```python
  client.query_points(
      collection_name,
      prefetch=[
          models.Prefetch(query=dense_vec, using="dense", limit=dense_top_k),
          models.Prefetch(query=sparse_vec, using="sparse", limit=sparse_top_k),
      ],
      query=models.FusionQuery(fusion=models.Fusion.RRF),
      limit=fused_top_k,
  )
  ```
  RRF (Reciprocal Rank Fusion) combine les deux classements sans avoir besoin
  de normaliser les scores entre dense et sparse — c'est la méthode par
  défaut recommandée par Qdrant pour démarrer. `DBSF` est une alternative
  plus sensible aux distributions de scores, à essayer seulement si RRF ne
  donne pas satisfaction.
- Payload : stocke le texte du chunk + toutes les métadonnées de provenance
  directement dans le `payload` de chaque point — Qdrant permet ensuite de
  filtrer (`Filter(...)`) par notebook/section en plus de la recherche
  hybride.
- IDs : Qdrant exige des entiers ou des UUID comme point id — génère un
  UUID5 déterministe à partir de ton `chunk_id` si celui-ci est une string
  arbitraire, pour pouvoir réindexer de façon idempotente.
- Doc hybrid search : https://qdrant.tech/documentation/concepts/hybrid-queries/

## 6. Reranking

La recherche vectorielle seule ramène souvent du bruit dans le top-10. Un
reranker cross-encoder réordonne un plus grand pool (ex: top-30) pour ne
garder que les meilleurs (ex: top-5) avant de les donner au LLM.

- **Qwen3-Reranker-0.6B** : modèle causal, scoring par probabilité
  yes/no sur un prompt spécifique — plus riche mais plus complexe à
  implémenter (voir la carte modèle HF pour le template exact)
- **BAAI/bge-reranker-v2-m3** : cross-encoder classique, utilisable
  directement avec `sentence_transformers.CrossEncoder(...).predict(pairs)`
  — recommandé pour une première version simple

## 7. Génération

- **Qwen2.5-1.5B-Instruct via `transformers`** (pas d'Ollama — jamais
  vraiment installé/configuré sur cette machine, alors que ce modèle était
  déjà en cache Hugging Face). `AutoModelForCausalLM` + `AutoTokenizer`,
  chargés une seule fois, avec `tokenizer.apply_chat_template(...)` pour
  construire le prompt (Qwen2.5-Instruct a un chat template intégré).
- Modèle petit (1.5B) : suffisant pour valider tout le pipeline de bout en
  bout, mais qualité de réponse limitée par rapport à un modèle plus gros —
  à améliorer plus tard si besoin (modèle Qwen plus gros, ou remonter Ollama
  pour changer de backend plus facilement — le code garde `backend:
  transformers | ollama` dans `config.yaml` pour cette raison).
- Prompt : instructions + contexte (chunks rerankés, avec leur source) +
  question. Demander explicitement au modèle de citer la page source
  (`[Source: Titre de page]`) et de dire "je ne sais pas" si le contexte ne
  contient pas la réponse (éviter les hallucinations)
- Attention à la taille du contexte : limiter le nombre de chunks selon la
  fenêtre de contexte du modèle (Qwen2.5-1.5B-Instruct : 32K tokens, large
  marge pour 5 chunks de quelques centaines de tokens chacun)

## 8. Évaluation — RAGAS

RAGAS mesure objectivement la qualité du pipeline plutôt que de juger "à
l'œil" quelques réponses. Il calcule des métriques à partir de 4 éléments par
question : `question`, `answer` (généré), `contexts` (chunks récupérés),
`ground_truth` (réponse de référence que **tu** écris).

- **Dataset d'évaluation** (`eval/qa_dataset.jsonl`) : à écrire toi-même,
  20-50 paires question/réponse représentatives de ce que tu voudras
  vraiment demander à ton RAG (définitions, comparaisons, "comment fonctionne
  X", questions nécessitant de croiser 2 pages...). C'est le travail le plus
  important de cette étape — un mauvais dataset de test donne une évaluation
  inutile, quel que soit le pipeline.
- **Métriques clés** (recommandées comme point de départ, voir `config.yaml`) :
  - `context_precision` / `context_recall` — évaluent la **récupération**
    (le hybrid search + reranking ramènent-ils les bons chunks ?)
  - `faithfulness` — la réponse générée est-elle fidèle au contexte fourni
    (pas d'hallucination) ?
  - `answer_relevancy` — la réponse répond-elle vraiment à la question posée ?
- **RAGAS a besoin d'un LLM juge** (pour faithfulness/relevancy) et d'un
  modèle d'embedding (pour certaines métriques de similarité). Par défaut
  RAGAS suppose OpenAI — il faut explicitement lui passer un wrapper
  LangChain pointant vers ton modèle local via `transformers`
  (`langchain_huggingface.HuggingFacePipeline`, enveloppé dans
  `ragas.llms.LangchainLLMWrapper`) et vers ton modèle d'embedding Qwen3,
  sinon ça tentera d'appeler l'API OpenAI et échouera sans clé.
- Le juge peut être le même modèle Qwen que pour la génération, ou un modèle
  plus gros si tu en as un disponible — un juge plus faible que le modèle
  évalué donne des scores moins fiables.
- Workflow : pour chaque question du dataset, faire tourner le pipeline
  complet (`retrieve_rerank.retrieve` + `generate.call_ollama`), collecter
  `contexts`/`answer`, construire un `datasets.Dataset` au format attendu par
  `ragas.evaluate(...)`, sauvegarder les scores dans `eval/results/`.
- Doc RAGAS : https://docs.ragas.io/

## Résumé des choix par défaut (modifiables dans `config.yaml`)

| Étape        | Choix par défaut            | Alternative                    |
|--------------|-------------------------------|----------------------------------|
| Extraction   | Graph API (device code)      | Export manuel HTML/DOCX         |
| Parsing      | BeautifulSoup + markdownify  | —                                 |
| Chunking     | Par heading + overlap        | Taille fixe en tokens            |
| Embeddings   | Qwen3-Embedding-0.6B (dense) | Qwen3-Embedding-4B/8B            |
| Sparse       | BM25 (fastembed)              | SPLADE                           |
| Vector store | Qdrant, hybrid (dense+sparse)| Chroma / LanceDB (dense only)    |
| Fusion       | RRF                            | DBSF                             |
| Reranker     | bge-reranker-v2-m3            | Qwen3-Reranker-0.6B              |
| Génération   | Qwen2.5-1.5B-Instruct (transformers) | Ollama (autre backend)    |
| Évaluation   | RAGAS (juge = Qwen2.5-1.5B)   | RAGAS (juge = OpenAI)            |

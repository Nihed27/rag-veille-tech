# RAG Veille Tech — Assistant multi-agents sur corpus de documents techniques

Assistant qui répond à des questions sur un corpus de documents techniques (ici : un cours d'Analyse de Données) en s'appuyant sur un pipeline RAG (Retrieval-Augmented Generation) et une orchestration multi-agents.

## Stack technique

- **Python**
- **LangChain + LangGraph** — orchestration multi-agents
- **ChromaDB** — base de vecteurs (stockage des embeddings)
- **LLM via OpenRouter** — génération de réponses
- **FastAPI** — exposition de l'API
- **Streamlit** — interface utilisateur
- **Docker / Docker Compose** — conteneurisation et déploiement local

## Architecture

### Pipeline d'ingestion (hors ligne, une fois)

```
PDF (cours) → découpage en chunks → embeddings → stockage dans ChromaDB
```

### Pipeline de question-réponse (à chaque requête)

```
Question utilisateur
        │
        ▼
┌───────────────┐
│   Retriever    │  cherche les passages pertinents dans ChromaDB
└───────┬───────┘
        ▼
┌───────────────┐
│   Rédacteur    │  formule une réponse à partir des passages trouvés
└───────┬───────┘
        ▼
┌───────────────┐
│    Critique    │  vérifie que la réponse est bien sourcée (verdict: ok / not ok)
└───────┬───────┘
        ▼
  Réponse + sources (document, page) + verdict
```

Ces trois agents sont orchestrés avec **LangGraph**, exposés via une route **FastAPI** (`POST /ask`), et consommés par une interface **Streamlit**.

### Déploiement

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   Conteneur Streamlit    │──── ▶│    Conteneur FastAPI     │
│   (interface web,        │ HTTP │   (pipeline RAG          │
│    port 8501)            │◀──── │    multi-agents,         │
│                          │      │    port 8000)            │
└─────────────────────────┘      └───────────┬─────────────┘
                                              ▼
                                       ChromaDB (volume local)
```

Les deux conteneurs sont lancés ensemble avec Docker Compose et communiquent via le réseau Docker interne.

## Installation et lancement

### Prérequis

- Docker et Docker Compose installés
- Une clé API [OpenRouter](https://openrouter.ai/)

### Configuration

Créer un fichier `.env` à la racine du projet :

```
OPENROUTER_API_KEY=ta_clé_ici
```

### Lancer le projet

```bash
docker compose up --build
```

Une fois les deux services démarrés :

- **Interface utilisateur (Streamlit)** : http://localhost:8501
- **API brute (FastAPI, documentation interactive)** : http://localhost:8000/docs

## Exemple

**Question :** *Qu'est-ce que l'ACP (analyse en composantes principales) ?*

**Réponse générée :**

> L'Analyse en Composantes Principales (ACP) a pour objectif de résumer un ensemble de données quantitatives. Ces données sont relatives à un grand nombre d'individus et/ou de variables illustrés dans un tableau à n lignes (chaque ligne représentant un individu de l'échantillon étudié composé de p colonnes). Elle permet notamment de décrire et représenter le réseau d'interaction entre les variables et les ressemblances entre les individus par rapport à l'ensemble des variables.

**Sources citées :**
- DA - Analyse en Composantes Principales (1).pdf — page 7
- DA - Analyse en Composantes Principales (1).pdf — page 36
- DA - Analyse en Composantes Principales (1).pdf — page 39
- DA - Analyse en Composantes Principales (1).pdf — page 8

**Verdict de l'agent critique :** `ok` — réponse validée comme bien sourcée.

## Structure du projet

```
rag-veille-tech/
├── app/
│   ├── config.py          # configuration (clé API, modèle, etc.)
│   ├── ingest.py           # ingestion et chunking des documents
│   ├── embeddings.py       # génération des embeddings
│   ├── vectorstore.py      # interface avec ChromaDB
│   ├── llm.py               # appel au LLM via OpenRouter
│   ├── graph.py             # orchestration multi-agents (LangGraph)
│   ├── rag.py               # logique du pipeline RAG
│   └── main.py               # point d'entrée FastAPI
├── data/                    # documents source + base ChromaDB (volume)
├── streamlit_app.py         # interface utilisateur
├── Dockerfile                # image de l'API
├── Dockerfile.streamlit      # image de l'interface
├── docker-compose.yml        # orchestration des deux services
├── requirements.txt           # dépendances de l'API
└── requirements-streamlit.txt # dépendances de l'interface

```

## Limites connues / scope

- Corpus fixe, chargé une fois (pas d'upload dynamique de documents)
- Pas d'authentification
- Un seul document source pour la démo actuelle (extensible à un corpus plus large)

## Pistes d'évolution

- Support de plusieurs documents / corpus dynamique
- Historique de conversation
- Déploiement de l'API sur un hébergeur cloud (Railway, Render...) pour un accès public complet

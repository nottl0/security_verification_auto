# SOC Offense AI-Summary Service

A small Python/FastAPI service that given a json offense
performs Retrieval-Augmented Generation (RAG) over an internal knowledge
base and returns a structured AI summary the analyst can act on immediately.

## What it does

1. **Accepts** a JSON offense payload (`POST /analyze`)
2. **Loads** the knowledge base documents from `knowledge_base/dynamic`, **chunks**
   them (based on structural components separated by ## ), **embeds** the chunks locally, and stores
   them in a **vector store** (Chroma) — built at service startup.
3. **Semantically searches** the vector store using a query built from the
   offense's rule description, attacker/target context, event categories,
   destination ports, and usernames involved.
4. **Builds a prompt** combining the offense data and the retrieved
   knowledge-base excerpts, together with files that one might consider always useful for making a 
   decision (knowledge_base/static/asset_classification.md).
5. **Calls the LLM** (Gemini API, because free tier is available) and returns a
   validated, structured response:
   - `summary` — human-readable description of what happened
   - `classification` — `TP` / `FP`, with
     `classification_reason`
   - `recommended_action` and `playbook_reference`
   - `confidence` (`High`/`Medium`/`Low`)
   - `supporting_context` — which KB chunks were used, with similarity scores

## Project layout

```
app/
  analyzer.py      System + user prompt construction and llm request
  main.py          FastAPI app (builds the vector store on startup, /analyze, /health)
  models.py        Pydantic schemas: Offense (input), AISummary (output)
  vector_store.py  Document loading, chunking, embedding, Chroma vector store and search
knowledge_base/    
  dynamic/         Files to search from dynamically given an offense
  static/          Files to always include in system prompt
examples/
  sample_offense.json   Offense sample
requirements.txt   Project requirements
test_script.py     Script to check user prompt (not system prompt) and chunks retrieved 
                   for a sample offense
```

## Setup

Generate a free-tier api key to prompt gemini api and add it as environmental variable

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

export GEMINI_API=...
```

## Running

### As a service

```bash
uvicorn app.main:app --reload --port 8000
```

On startup the service builds the vector store from `knowledge_base/dynamic`. Send requests using:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @examples/sample_offense.json
```

Interactive API docs: http://localhost:8000/docs

## Notes

- **Chunking** is a simple, dependency-free paragraph-aware splitter with
  configurable size/overlap — adequate for the structured markdown
  playbooks used here; for larger/heterogeneous corpora a
  recursive/semantic chunker (e.g. via `langchain-text-splitters`) would be
  a natural upgrade.

- **LLM output is TP by default**: the pipeline parses the
  model's JSON and falls back to safe defaults (`TP` /
  `Low` confidence) if the model returns an unexpected classification or
  confidence value, so it would have to be checked in any case.

## Test that retrieval works with

```bash
python test_script.py
```
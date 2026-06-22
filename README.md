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

Generate a free-tier api key to prompt gemini api (https://aistudio.google.com/api-keys) and add it as environmental variable

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

export GEMINI_API_KEY=...
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

- **Chunking** utilizes a hybrid semantic pipeline via langchain-text-splitters tailored 
  for markdown playbooks. It first fragments files structurally on # and ## header
  boundaries (MarkdownHeaderTextSplitter), then refines those sections with a 
  character-aware budget (RecursiveCharacterTextSplitter). 
  This enforces strict size limits and carries path/header context down into every 
  sub-chunk while cleanly preventing the formation of tiny, isolated header fragments.

- **Embedding** uses paraphrase-multilingual-mpnet-base-v2 which is a multilingual model  and should
  work better for russian-enlgish cross lingual embedding

- **LLM output is TP by default**: the pipeline parses the
  model's JSON and falls back to safe defaults (`TP` /
  `Low` confidence) if the model returns an unexpected classification or
  confidence value, so it would have to be checked in any case.

## Test LLM output

The output of the LLM and retrieved relevant chunks can be tested using `test_script.py`. There are two examples in the /examples folder one for tp and one for fp offense. Change the name of the json file used in `test_script.py` to see the change in the response.


```bash
python test_script.py
```

## Possible improvements 

- Some domain knowledge would be best to restructure the json of the offense accepted by the service into natural language. As an example, the ip could have a tag indicating if its internal, so that some patterns can be mathced better (Source будет `внутренним` IP из инфраструктурного диапазона.)

- If the documents in the knowledge base can be translated to english it would possibly boost the quality of similarity search (since json body has english words)
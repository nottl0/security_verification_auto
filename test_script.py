import json
from app.vector_store import build_vector_store, search
from app.models import Offense
from app.analyzer import build_user_prompt

build_vector_store()

offense = Offense(**json.load(open('examples/sample_offense.json')))

query = offense.to_search_query()
print('Search query:', query)
print()

chunks = search(query)
prompt = build_user_prompt(offense=offense, chunks=chunks)
print(prompt)

for i, chunk in enumerate(chunks, 1):
    print(f'--- Result {i} | source: {chunk.source} | score: {chunk.score} | id: {chunk.chunk_id}')
    print(chunk.text)
    print()
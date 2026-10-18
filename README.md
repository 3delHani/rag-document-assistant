# AI Document Assistant (RAG + FastAPI)

Upload a text document and ask questions about it. The app splits the document into chunks, embeds them,
retrieves the most relevant chunks for each question, and asks an LLM to answer **only from that context**.
If the answer is not in the document, the API says so explicitly.

## How it works

```
Upload .txt/.md ──► split into overlapping chunks ──► embed (sentence-transformers) ──► store (JSON + .npy)

Question ──► embed ──► cosine-similarity top-K chunks ──► LLM answers from context only ──► answer + sources
```
## Installation

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd rag-document-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env and set OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.
(The first start downloads the embedding model, ~90 MB.)

## Run with Docker

```bash
cp .env.example .env             # set OPENAI_API_KEY
docker compose up --build
```

The API is then available at http://127.0.0.1:8000/docs. Uploaded documents are stored in `./data`
(mounted as a volume), so they survive container restarts.

Or without compose:

```bash
docker build -t rag-document-assistant .
docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" rag-document-assistant
```

The embedding model is downloaded at build time, so the container starts quickly. If you change
`EMBEDDING_MODEL`, rebuild with `docker compose build --build-arg EMBEDDING_MODEL=<model>`.
When using Ollama on your host machine, set `OPENAI_BASE_URL=http://host.docker.internal:11434/v1`.

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents` | Upload a `.txt`/`.md` file → returns `document_id` |
| GET | `/documents` | List uploaded documents |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/ask` | Ask a question about a document |
| GET | `/health` | Health check |

### Quick start with curl

```bash
# 1) upload the sample document
curl -F "file=@sample_docs/company_handbook.txt" http://127.0.0.1:8000/documents
# -> {"document_id":"3f9c2a1b7d4e","filename":"company_handbook.txt","num_chunks":4}

# 2) ask a question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"document_id":"3f9c2a1b7d4e","question":"How many days of annual leave do employees get?"}'
```

## Example questions (using `sample_docs/company_handbook.txt`)

LLM wording may vary slightly; the facts should match.

**1. "How many days of annual leave do employees get?"**
```json
{
  "answer": "Full-time employees get 21 days of paid annual leave per year (30 days after more than 10 years of service).",
  "answer_found": true,
  "sources": [{"chunk_index": 0, "score": 0.7, "text": "Nile Tech Employee Handbook Annual Leave Every full-time employee..."}]
}
```

**2. "What is the remote work policy?"**
```json
{
  "answer": "Employees can work remotely up to two days per week with manager approval, and must be available on Slack from 10:00 to 16:00 Cairo time. Fully remote work is only for remote-eligible roles.",
  "answer_found": true,
  "sources": [{"chunk_index": 1, "score": 0.6, "text": "Remote Work Policy Employees may work remotely up to two days per week..."}]
}
```

**3. "Who won the 2022 World Cup?"** (not in the document)
```json
{
  "answer": "The requested information is not available in the document.",
  "answer_found": false,
  "sources": []
}
```

(Scores shown are illustrative.)

## Configuration

All settings live in `.env` (see `.env.example`): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`,
`EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SIMILARITY`.

To use a local model through Ollama: `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`, `LLM_MODEL=llama3.1`.

## Tests

```bash
pytest -q
```

## Project structure

```
app/
  main.py          FastAPI routes + RAG flow
  chunking.py      sentence-aware overlapping splitter
  embeddings.py    sentence-transformers wrapper
  vector_store.py  file-backed store + cosine search
  llm.py           grounded-answer prompt + OpenAI-compatible client
  schemas.py       Pydantic models
  config.py        environment-based settings
Dockerfile, docker-compose.yml, .dockerignore
sample_docs/       example document
tests/             unit tests
```

## Limitations / ideas for improvement

- Supports UTF-8 `.txt` and `.md` only (PDF/DOCX parsing would be a natural next step).
- Embedding model is English-focused; for Arabic documents use `paraphrase-multilingual-MiniLM-L12-v2`.
- Swap the NumPy store for FAISS/Chroma/pgvector when handling many large documents.
- Add re-ranking and streaming responses for better quality and UX.

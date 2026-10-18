from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import config
from app.chunking import split_text
from app.embeddings import embed_texts, warm_up
from app.llm import NOT_FOUND_TOKEN, generate_answer
from app.schemas import AskRequest, AskResponse, Source, UploadResponse
from app.vector_store import DocumentStore

store = DocumentStore(config.DATA_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()  # load the embedding model once at startup
    yield


app = FastAPI(
    title="AI Document Assistant (RAG)",
    description="Upload a text document and ask questions about it.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith((".txt", ".md")):
        raise HTTPException(400, "Only .txt and .md files are supported.")

    raw = file.file.read()
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded text.")

    chunks = split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(400, "The document is empty.")

    meta = store.add(file.filename, chunks, embed_texts(chunks))
    return UploadResponse(
        document_id=meta["document_id"],
        filename=meta["filename"],
        num_chunks=meta["num_chunks"],
    )


@app.get("/documents")
def list_documents():
    return store.list()


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    if not store.delete(document_id):
        raise HTTPException(404, "Document not found.")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if store.get(req.document_id) is None:
        raise HTTPException(404, "Document not found. Upload it first via POST /documents.")

    query_vec = embed_texts([req.question])[0]
    hits = store.search(req.document_id, query_vec, req.top_k)
    sources = [Source(chunk_index=i, score=round(s, 4), text=t) for i, t, s in hits]

    # Guard 1: retrieval found nothing even remotely related -> don't call the LLM.
    if not hits or hits[0][2] < config.MIN_SIMILARITY:
        return AskResponse(answer=config.NOT_FOUND_MESSAGE, answer_found=False, sources=[])

    # Guard 2: the LLM is instructed to say NOT_FOUND if the context lacks the answer.
    try:
        answer = generate_answer(req.question, [t for _, t, _ in hits])
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")

    if NOT_FOUND_TOKEN in answer:
        return AskResponse(answer=config.NOT_FOUND_MESSAGE, answer_found=False, sources=[])

    return AskResponse(answer=answer, answer_found=True, sources=sources)

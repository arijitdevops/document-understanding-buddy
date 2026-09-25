# Document Understanding Buddy

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![MySQL](https://img.shields.io/badge/MySQL-8-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A GenAI chat app that helps you **understand the documents you upload into the conversation**.
Drop a PDF, Word file, slide deck, CSV, Markdown/text file or an image into a chat, then ask
questions, request a summary, an "explain like I'm new" walkthrough, the key points or a
glossary. Every answer is **grounded in your documents** (Retrieval-Augmented Generation) and
carries **citation chips** that show the exact passage, document, page and chunk it came from.

Built with **Google Gemini** (`gemini-2.5-flash` + `gemini-embedding-001`), **LangChain** as the
LLM framework, a local **Chroma** vector store, **FastAPI**, **MySQL** and a **React + TypeScript** UI.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
  - [Option A: Docker Compose (everything)](#option-a-docker-compose-everything)
  - [Option B: Local development on macOS / Linux](#option-b-local-development-on-macos--linux)
  - [Option C: Local development on Windows](#option-c-local-development-on-windows-powershell)
- [Configuration](#configuration)
- [Usage](#usage)
- [API reference](#api-reference)
- [Testing](#testing)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [License](#license)

---

## Features

- **Chat sessions**: create, rename and delete chats; the full conversation history is persisted
  in MySQL and reloaded when you reopen a chat.
- **Upload documents inside the chat**: drag and drop onto the conversation or use the paperclip.
  Supported: **PDF, DOCX, PPTX, TXT, MD, CSV** and **PNG/JPG/WEBP images** (transcribed and
  described with Gemini vision). Up to 10 files per upload, 25 MB each by default.
- **Background ingestion with live status**: each file goes `pending → parsing → chunking →
  embedding → ready` (or `failed` with a readable reason). The document sidebar shows a
  progress bar and the current stage; failed files can be retried.
- **Grounded answers with citations**: answers use only the chat's documents. Inline `[n]`
  markers become clickable chips that show the source snippet, file name, page and chunk,
  with a "Show full passage" option. Citation numbers the model invents are removed
  server-side before the answer is saved.
- **Document understanding tools** (one click per document, or across several):
  - **Summarize**: overview plus key bullets
  - **Explain simply** (explain like I'm new): plain language, terms defined
  - **Key points**: ranked list with a bottom line
  - **Glossary**: a table of terms, meanings and sources
  - **Compare**: agreements, differences and what only one document covers
- **Multi-document question answering** scoped to the current chat's documents. Tick documents
  in the sidebar to narrow the scope further. Documents from other chats are never searched.
- **Hybrid retrieval**: dense vectors (Gemini embeddings in Chroma) plus BM25 keyword search,
  fused with Reciprocal Rank Fusion. An optional LLM reranker is available.
- **Streaming responses** over Server-Sent Events (SSE), token by token.
- **Delete documents**: removes the MySQL rows, the Chroma vectors and the stored file.
- **Graceful degradation**: the API starts without `GEMINI_API_KEY`; `GET /api/health`
  reports the missing key by name and the UI shows a banner explaining what to do.
- **Prompt-injection hygiene**: document text is fenced as untrusted data, and passages
  containing instruction-like phrases are flagged.

## Architecture

### Ingestion (upload → searchable)

```mermaid
flowchart LR
    U[React UI<br/>drag and drop] -->|multipart POST<br/>/api/sessions/:id/documents| API[FastAPI]
    API -->|validate type/size,<br/>SHA-256 dedupe| FS[(Upload folder)]
    API -->|documents +<br/>ingestion_jobs rows| DB[(MySQL)]
    API -. BackgroundTasks .-> P[Ingestion pipeline]
    P --> L{Loader by type}
    L -->|pypdf + pdfplumber| T[Pages of text]
    L -->|python-docx / python-pptx / csv / text| T
    L -->|images: Gemini vision<br/>via LangChain| T
    T --> C[Recursive chunker<br/>page + heading provenance]
    C -->|chunk rows| DB
    C -->|Chroma.add_texts in batches| E[GoogleGenerativeAIEmbeddings<br/>gemini-embedding-001<br/>RETRIEVAL_DOCUMENT]
    E --> V[(Chroma<br/>one collection per chat)]
    P -->|status + progress| DB
    U -. polls status .-> API
```

### Retrieval and answering (question → cited, streamed answer)

```mermaid
flowchart LR
    Q[User question<br/>or task] --> API[POST /api/sessions/:id/chat]
    API --> H[(MySQL:<br/>chat history)]
    API --> K{Task}
    K -->|qa| R1[Dense search<br/>Chroma similarity<br/>RETRIEVAL_QUERY]
    K -->|qa| R2[BM25 keyword search]
    R1 --> F[Reciprocal Rank Fusion]
    R2 --> F
    F --> RR[Optional LLM rerank]
    K -->|summarize / eli5 / key points /<br/>glossary / compare| D[Whole-document context<br/>chunks in reading order]
    RR --> CTX[Numbered, fenced<br/>context sources]
    D --> CTX
    CTX --> LCEL[LCEL chain:<br/>ChatPromptTemplate → ChatGoogleGenerativeAI<br/>gemini-2.5-flash → StrOutputParser]
    H --> LCEL
    LCEL -->|SSE tokens| UI[React UI]
    LCEL --> VC[Verify citations<br/>drop invented markers]
    VC -->|citations event| UI
    VC -->|assistant message + query log| H
```

**Key design decisions**

| Decision | Why |
| --- | --- |
| Documents belong to a chat session | "Ask about what I uploaded here" is the core interaction; scoping, deletion and history all follow the chat. |
| One Chroma collection per chat, `document_id` in metadata | Chat-level isolation, cheap per-document filtering and deletion. |
| Hybrid dense + BM25 with RRF | Embeddings miss rare exact tokens (invoice numbers, names, error codes); BM25 catches them. RRF needs no score calibration. |
| Whole-document context for summarize/explain/key points/glossary/compare | Similarity search with a topic-less request ("summarize") returns arbitrary chunks. These tasks read chunks in order, sampled evenly from start to end when the document exceeds `DOCUMENT_TASK_MAX_CHARS`. |
| Citation verification after generation | Models sometimes cite `[7]` when only five sources exist; such markers are removed and only real sources are returned. |
| Chunk rows in MySQL and vectors in Chroma | MySQL is the source of truth for chunk text (used for citations and document tasks); Chroma is the search index. |

## Tech stack

| Layer | Technology |
| --- | --- |
| LLM | Google Gemini `gemini-2.5-flash` (chat and vision), `gemini-embedding-001` (embeddings) |
| LLM framework | LangChain: `langchain-google-genai` (`ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`), `langchain-chroma`, LCEL runnables |
| Vector store | Chroma (persistent, local directory) |
| Backend | Python 3.11+, FastAPI, SQLAlchemy 2 (async), asyncmy, Pydantic Settings |
| Database | MySQL 8 (SQLite via aiosqlite for tests or a quick local run) |
| Parsing | pypdf, pdfplumber (tables), python-docx, python-pptx, csv, Gemini vision for images |
| Frontend | React 19, TypeScript, Vite, react-markdown + remark-gfm |
| Tests | pytest + pytest-asyncio (fake Gemini, SQLite), Vitest |
| Deployment | Docker Compose (MySQL + backend + nginx-served frontend) |

## Project structure

```
document-understanding-buddy/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app factory, lifespan, CORS
│   │   ├── config.py             # Settings from environment / .env
│   │   ├── logging_config.py
│   │   ├── api/
│   │   │   ├── router.py         # Mounts all routes under /api
│   │   │   ├── deps.py           # DB session, lookups, rate limiting
│   │   │   ├── errors.py         # Uniform {"detail", "code"} errors
│   │   │   └── routes/           # health, sessions, documents, chat (SSE)
│   │   ├── db/                   # Declarative base, async engine, ORM models
│   │   ├── ingestion/            # loaders, chunker, jobs, pipeline
│   │   ├── rag/
│   │   │   ├── vectorstore.py    # langchain-chroma wrapper, one collection per chat
│   │   │   ├── retriever.py      # Dense + BM25 hybrid retrieval with RRF
│   │   │   ├── reranker.py       # Optional LCEL reranker
│   │   │   ├── document_context.py # Whole-document context for document tasks
│   │   │   ├── prompts.py        # System prompt, task templates, injection scan
│   │   │   └── chain.py          # LCEL RAG chain, streaming, citation verification
│   │   ├── schemas/api.py        # Pydantic request/response models
│   │   └── services/             # gemini.py (LangChain models), storage, rate limit
│   ├── tests/                    # pytest suite with fake Gemini + SQLite
│   ├── schema.sql                # MySQL 8 schema (same as auto-created tables)
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── pytest.ini
│   ├── ruff.toml
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # State, streaming, polling
│   │   ├── api.ts                # REST client + SSE chat stream
│   │   ├── sse.ts                # Incremental SSE parser
│   │   ├── citations.ts          # [n] markers → citation chips
│   │   ├── components/           # Sidebar, ChatView, MessageBubble, CitationChip, DocumentPanel, HealthBanner
│   │   └── *.test.ts             # Vitest unit tests
│   ├── nginx.conf                # Production proxy (SSE-friendly)
│   ├── package.json / package-lock.json
│   └── Dockerfile
├── samples/
│   └── home-solar-guide.md       # Small document to try the app with
├── docker-compose.yml
├── .env.example
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites

- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)
- For Docker: Docker Desktop (or Docker Engine) with Compose v2
- For local development: **Python 3.11+**, **Node.js 20.19+** (22 LTS recommended) and
  **MySQL 8** (the Compose file can run MySQL for you)

### Option A: Docker Compose (everything)

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
# edit .env and set GEMINI_API_KEY=...
docker compose up --build
```

- UI: <http://localhost:8080>
- API docs (Swagger): <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

MySQL data, uploads and the Chroma index are kept in named Docker volumes
(`mysql-data`, `backend-data`). `docker compose down -v` removes them.

### Option B: Local development on macOS / Linux

```bash
# 1. Configuration (from the repository root)
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...

# 2. MySQL: start only the database container...
docker compose up -d mysql
#    ...or use your own MySQL 8 server:
#    mysql -u root -p < backend/schema.sql
#    and point DATABASE_URL in .env at it

# 3. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# 4. Frontend (in a second terminal)
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` to `http://localhost:8000`
(override with `VITE_API_PROXY`).

> **No MySQL at hand?** Set `DATABASE_URL=sqlite+aiosqlite:///./data/doc_buddy.db` in `.env`
> for a quick single-user run. MySQL remains the supported database.

### Option C: Local development on Windows (PowerShell)

```powershell
# 1. Configuration (from the repository root)
Copy-Item .env.example .env
notepad .env            # set GEMINI_API_KEY=...

# 2. MySQL: start only the database container (Docker Desktop)...
docker compose up -d mysql
#    ...or install MySQL 8 and run:  Get-Content backend\schema.sql | mysql -u root -p

# 3. Backend
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# 4. Frontend (in a second PowerShell window)
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

## Configuration

All settings are environment variables (case-insensitive). The backend reads `.env` from its
working directory and from the parent directory, so a single `.env` in the repository root
works both for Docker Compose and for running `uvicorn` inside `backend/`.

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(none)* | **Required** for ingestion and answers. Without it the API still starts and `/api/health` lists it under `missing_config`. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Chat model used for answers, document tasks and reranking. |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model for chunks and queries. Changing it requires re-ingesting documents. |
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Model used to transcribe uploaded images. |
| `GEMINI_TEMPERATURE` | `0.2` | Sampling temperature. |
| `GEMINI_MAX_RETRIES` | `3` | Retries LangChain performs on transient Gemini errors. |
| `GEMINI_TIMEOUT_SECONDS` | `120` | Per-request timeout. |
| `DATABASE_URL` | `mysql+asyncmy://doc_buddy:doc_buddy@localhost:3306/doc_buddy` | SQLAlchemy async URL. |
| `AUTO_CREATE_TABLES` | `true` | Create missing tables on startup (same DDL as `backend/schema.sql`). |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Chroma directory (relative paths resolve against `backend/`). |
| `UPLOAD_DIR` | `./data/uploads` | Where uploaded files are stored. |
| `MAX_UPLOAD_MB` | `25` | Per-file size limit. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters per chunk and overlap. |
| `RETRIEVAL_TOP_K` | `6` | Passages given to the model for a question. |
| `RETRIEVAL_CANDIDATES` | `30` | Candidates fetched from each retriever before fusion. |
| `HYBRID_ENABLED` / `HYBRID_ALPHA` | `true` / `0.5` | Enable BM25 fusion; alpha 1.0 = dense only, 0.0 = BM25 only. |
| `RERANK_ENABLED` | `false` | LLM reranking of fused candidates (one extra Gemini call per question). |
| `HISTORY_MESSAGES` | `6` | Earlier messages sent with each question for follow-ups. |
| `DOCUMENT_TASK_MAX_CHARS` | `60000` | Context budget for summarize / explain / key points / glossary / compare. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Chat requests per client IP per minute. |
| `LOG_LEVEL` / `ENVIRONMENT` | `INFO` / `development` | `development` logs plain text, anything else logs JSON. |
| `VITE_API_BASE_URL` *(frontend, build time)* | *(empty)* | Absolute API origin if the UI is not served behind the same host as `/api`. |

## Usage

1. Click **+ New chat** (or just drop a file; a chat is created for you). To try the app
   quickly, upload `samples/home-solar-guide.md` and ask "How many panels do I need for
   9,000 kWh a year?" or click **Glossary**.
2. Drag files onto the conversation, or click the paperclip. Watch the **Documents** panel:
   each file shows its stage and a progress bar until it is **Ready**.
3. Ask a question. The answer streams in; `[n]` chips link to the supporting passages.
   Click a chip to see the snippet, document, page and chunk; click **Show full passage**
   for the whole chunk. A **Sources** row under each answer lists every cited passage.
4. Use the buttons on a document card: **Summarize**, **Explain simply**, **Key points**,
   **Glossary**. With two or more ready documents, **Summarize all**, **Compare** and
   **Combined glossary** work across them.
5. Tick documents to limit questions to them; untick all to search every ready document
   in the chat.
6. **Delete** removes a document (rows, vectors and file). Deleting a chat removes its
   history and all of its documents.

## API reference

Base URL: `http://localhost:8000/api`. Interactive docs are at `/docs`. Errors are returned as
`{"detail": "...", "code": "..."}`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Database, vector store and Gemini status; `missing_config` lists absent settings (for example `["GEMINI_API_KEY"]`). Always HTTP 200. |
| `GET` | `/sessions` | List chats (most recent first) with document and message counts. |
| `POST` | `/sessions` | Create a chat. Body (optional): `{"title": "..."}`. Returns 201. |
| `GET` | `/sessions/{id}` | Chat with its `documents` (including status) and `messages` (including citations). |
| `PATCH` | `/sessions/{id}` | Rename: `{"title": "..."}`. |
| `DELETE` | `/sessions/{id}` | Delete chat, history, documents, vectors and files. Returns 204. |
| `GET` | `/sessions/{id}/messages` | Conversation history, oldest first. |
| `GET` | `/sessions/{id}/documents` | Documents in the chat with `status`, `progress` (0 to 1), `stage_detail`, `error`. |
| `POST` | `/sessions/{id}/documents` | Upload one or more files (multipart field `files`). Returns 202 with `documents`, `duplicates` and `rejected`. Ingestion runs in the background. |
| `GET` | `/documents/{id}` | One document and its status. |
| `GET` | `/documents/{id}/job` | Latest ingestion job (`status`, `progress`, `stage_detail`, timestamps). |
| `POST` | `/documents/{id}/reingest` | Re-run ingestion (for example after adding the API key). Returns 202. |
| `DELETE` | `/documents/{id}` | Delete a document, its chunks, vectors and file. Returns 204. |
| `GET` | `/chunks/{id}` | Full text, page and heading of a cited chunk. |
| `POST` | `/sessions/{id}/chat` | Ask a question or run a task; streams **Server-Sent Events**. |

### Chat request

```json
{
  "message": "What does the warranty cover?",
  "task": "qa",
  "document_ids": ["optional", "subset", "of", "this chat's documents"]
}
```

`task` is one of `qa` (default; hybrid retrieval), `summarize`, `eli5`, `key_points`,
`glossary`, `compare` (these read whole documents). `message` may be empty for the document
tasks. Error responses before streaming starts: `404` unknown chat, `400 no_documents`,
`409 documents_processing`, `422 empty_message`, `429 rate_limited`,
`503 gemini_not_configured`.

### SSE events

| Event | Payload |
| --- | --- |
| `meta` | `{"session_id", "user_message_id", "task", "title"}` |
| `sources` | `{"task", "hybrid", "reranked", "retrieval_ms", "injection_flagged", "injection_notes", "sources": [{"marker", "document_id", "document_name", "page", "chunk_id", "chunk_index", "snippet"}]}` |
| `token` | `{"text": "..."}` (repeated) |
| `citations` | `{"answer": "<final text>", "citations": [{"marker", "document_id", "document_name", "page", "chunk_id", "chunk_index", "snippet"}]}` |
| `done` | `{"message_id", "retrieval_ms", "generation_ms", "model"}` |
| `error` | `{"message", "detail"}` (terminal) |

### Examples (curl)

```bash
SID=$(curl -s -X POST localhost:8000/api/sessions -H 'Content-Type: application/json' -d '{}' \
      | python -c "import sys, json; print(json.load(sys.stdin)['id'])")

curl -F "files=@report.pdf" -F "files=@notes.docx" localhost:8000/api/sessions/$SID/documents
curl localhost:8000/api/sessions/$SID/documents        # poll until status == "ready"

curl -N -X POST localhost:8000/api/sessions/$SID/chat \
     -H 'Content-Type: application/json' \
     -d '{"message": "What are the payment terms?"}'

curl -N -X POST localhost:8000/api/sessions/$SID/chat \
     -H 'Content-Type: application/json' -d '{"task": "eli5"}'
```

## Testing

The backend suite runs fully offline: Gemini is replaced by a scripted LangChain chat model and
a deterministic bag-of-words hashing embedding (`backend/tests/fakes.py`), the database is a
temporary SQLite file and Chroma writes to a temporary directory. The API tests exercise the
real FastAPI app, background ingestion, the langchain-chroma store, hybrid retrieval, SSE
streaming, citation verification and persistence.

```bash
# Backend
cd backend
pytest                    # all tests
ruff check app tests      # lint

# Frontend
cd frontend
npm test                  # Vitest: SSE parser, citation linking, formatting
npm run build             # type-check (tsc) + production build
```

## How it works

1. **Upload.** Files are validated against an extension allowlist, streamed to disk with a
   size cap, hashed (SHA-256) so an identical file is not ingested twice in the same chat,
   and recorded as `documents` rows with an `ingestion_jobs` row.
2. **Parse.** PDFs are read page by page with pypdf; pages that look like tables are
   re-read with pdfplumber. DOCX headings become Markdown headings, PPTX gives one page per
   slide (with speaker notes), CSVs are paginated every 200 rows, and images are transcribed
   and described by Gemini vision through LangChain.
3. **Chunk.** A recursive splitter (paragraph → line → sentence → word) keeps sentences intact
   and records each chunk's page and heading trail (for example `Pricing > Enterprise`).
4. **Embed and index.** Chunks are stored in MySQL and added to the chat's Chroma collection
   via `Chroma.add_texts`, which embeds them with `GoogleGenerativeAIEmbeddings`
   (`RETRIEVAL_DOCUMENT` task type). Batches report progress to the job row.
5. **Retrieve.** A question is embedded with the `RETRIEVAL_QUERY` task type for dense
   search, BM25 scores the same chunks lexically, and Reciprocal Rank Fusion merges both
   rankings. Optionally, Gemini reranks the fused candidates.
6. **Generate.** The LCEL chain renders numbered sources inside a fenced "untrusted data"
   block together with recent chat history and streams the answer from
   `ChatGoogleGenerativeAI`.
7. **Verify and store.** Citation markers that do not match a supplied source are removed;
   the answer, its citations and timing telemetry (`query_logs`) are saved to MySQL.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Yellow banner "Missing configuration: GEMINI_API_KEY" | Put the key in `.env` (repository root) and restart the backend. Then click **Retry** on failed documents. |
| `Can't connect to MySQL server` in the logs / health shows `database` not ok | Start MySQL (`docker compose up -d mysql`) and check `DATABASE_URL`. The API keeps running and recovers once the database is reachable. |
| Document stuck as **Failed** with "No readable text was extracted" | The PDF is a scan without a text layer. Export the pages as PNG/JPG and upload the images; Gemini vision transcribes them. |
| `429 rate_limited` | Raise `RATE_LIMIT_PER_MINUTE`, or wait a minute. Gemini free-tier quotas may also return errors on heavy use. |
| Answers look unrelated after changing `GEMINI_EMBEDDING_MODEL` | Vectors from different embedding models are not comparable. Re-ingest the documents or start a new chat. |
| Tokens arrive all at once behind a reverse proxy | Disable response buffering for `/api/` (see `frontend/nginx.conf`: `proxy_buffering off`). |

## Limitations

- The rate limiter and background ingestion run in-process: this is designed for a single
  API process. For several workers, move ingestion to a task queue and rate limiting to Redis.
- There is no user authentication; every visitor sees every chat. Add auth before exposing the
  app beyond your machine or network.
- `.doc`, `.xls(x)` and scanned PDFs are not parsed directly (convert to DOCX/CSV/images).
- Answer quality depends on the model and on the documents; always check the cited passages.

## License

[MIT](LICENSE) © 2026 Common Labs

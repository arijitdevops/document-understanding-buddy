-- Document Understanding Buddy -- MySQL 8 schema.
-- Equivalent to what the API creates on startup when AUTO_CREATE_TABLES=true.
-- Apply once to an empty server:  mysql -u root -p < backend/schema.sql
-- (docker compose mounts this file into MySQL's init directory automatically.)

CREATE DATABASE IF NOT EXISTS doc_buddy CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE doc_buddy;

CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(32) NOT NULL,
    title VARCHAR(200) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT now(),
    updated_at DATETIME NOT NULL DEFAULT now(),
    CONSTRAINT pk_chat_sessions PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS query_logs (
    id VARCHAR(32) NOT NULL,
    session_id VARCHAR(32),
    task VARCHAR(32) NOT NULL,
    question TEXT NOT NULL,
    model VARCHAR(80) NOT NULL,
    retrieved_chunk_ids JSON,
    retrieval_ms INTEGER NOT NULL,
    rerank_ms INTEGER NOT NULL,
    generation_ms INTEGER NOT NULL,
    citation_count INTEGER NOT NULL,
    injection_flagged INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT now(),
    CONSTRAINT pk_query_logs PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX ix_query_logs_created ON query_logs (created_at);

CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(32) NOT NULL,
    session_id VARCHAR(32) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    stored_name VARCHAR(255) NOT NULL,
    mime VARCHAR(128) NOT NULL,
    size_bytes BIGINT NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    status VARCHAR(16) NOT NULL,
    error TEXT,
    checksum VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT now(),
    updated_at DATETIME NOT NULL DEFAULT now(),
    CONSTRAINT pk_documents PRIMARY KEY (id),
    CONSTRAINT uq_documents_session_checksum UNIQUE (session_id, checksum),
    CONSTRAINT fk_documents_session_id_chat_sessions FOREIGN KEY(session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE,
    CONSTRAINT uq_documents_stored_name UNIQUE (stored_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX ix_documents_session_status ON documents (session_id, status);

CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(32) NOT NULL,
    session_id VARCHAR(32) NOT NULL,
    `role` VARCHAR(16) NOT NULL,
    content LONGTEXT NOT NULL,
    task VARCHAR(32) NOT NULL,
    citations JSON,
    document_ids JSON,
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT pk_messages PRIMARY KEY (id),
    CONSTRAINT fk_messages_session_id_chat_sessions FOREIGN KEY(session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX ix_messages_session_created ON messages (session_id, created_at);

CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(32) NOT NULL,
    document_id VARCHAR(32) NOT NULL,
    chunk_index INTEGER NOT NULL,
    page INTEGER,
    heading VARCHAR(500),
    content LONGTEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT now(),
    CONSTRAINT pk_document_chunks PRIMARY KEY (id),
    CONSTRAINT uq_document_chunks_document_index UNIQUE (document_id, chunk_index),
    CONSTRAINT fk_document_chunks_document_id_documents FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id VARCHAR(32) NOT NULL,
    document_id VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    progress FLOAT NOT NULL,
    stage_detail VARCHAR(255),
    error TEXT,
    started_at DATETIME,
    finished_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT now(),
    updated_at DATETIME NOT NULL DEFAULT now(),
    CONSTRAINT pk_ingestion_jobs PRIMARY KEY (id),
    CONSTRAINT fk_ingestion_jobs_document_id_documents FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX ix_ingestion_jobs_document_status ON ingestion_jobs (document_id, status);

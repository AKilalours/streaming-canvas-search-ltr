-- ============================================================
-- StreamLens — Production Database Schema
-- Akila Lourdes Miriyala Francis · MS in Artificial Intelligence
-- ============================================================
-- Supports: PostgreSQL 14+ / SQLite 3.35+
-- Usage:
--   psql -U streamlens -d streamlens_db -f schema.sql
--   sqlite3 streamlens.db < schema.sql
-- ============================================================

-- ── Users ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id         VARCHAR(64)     PRIMARY KEY,
    profile_name    VARCHAR(128)    NOT NULL,
    profile_type    VARCHAR(32)     NOT NULL DEFAULT 'standard',  -- chrisen | gilbert | alex
    language        VARCHAR(64)     NOT NULL DEFAULT 'English',
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    interaction_count INTEGER       NOT NULL DEFAULT 0,
    taste_breadth   FLOAT           DEFAULT NULL,   -- PySpark computed: genre diversity score
    watch_count     INTEGER         NOT NULL DEFAULT 0,
    cold_start      BOOLEAN         NOT NULL DEFAULT TRUE  -- True until >= 5 interactions
);

CREATE INDEX IF NOT EXISTS idx_users_profile_type ON users (profile_type);
CREATE INDEX IF NOT EXISTS idx_users_language     ON users (language);

-- ── Items (Movies / Content) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS items (
    doc_id          VARCHAR(64)     PRIMARY KEY,
    title           VARCHAR(512)    NOT NULL,
    year            SMALLINT        DEFAULT NULL,
    genres          TEXT            DEFAULT NULL,   -- comma-separated: "Crime,Drama,Thriller"
    tags            TEXT            DEFAULT NULL,   -- comma-separated: "cult film,twist"
    language        VARCHAR(64)     NOT NULL DEFAULT 'English',
    item_popularity FLOAT           NOT NULL DEFAULT 0.0,  -- normalised co-watch popularity
    rating_count    INTEGER         NOT NULL DEFAULT 0,
    avg_rating      FLOAT           DEFAULT NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_items_genres   ON items (genres);
CREATE INDEX IF NOT EXISTS idx_items_year     ON items (year);
CREATE INDEX IF NOT EXISTS idx_items_language ON items (language);

-- ── Ratings (33.8M MovieLens) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS ratings (
    rating_id       BIGSERIAL       PRIMARY KEY,
    user_id         VARCHAR(64)     NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id          VARCHAR(64)     NOT NULL REFERENCES items(doc_id)  ON DELETE CASCADE,
    rating          FLOAT           NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    timestamp       BIGINT          NOT NULL,  -- Unix timestamp (MovieLens format)
    rated_at        TIMESTAMP       GENERATED ALWAYS AS (
                        TO_TIMESTAMP(timestamp)
                    ) STORED,
    UNIQUE (user_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user_id  ON ratings (user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_doc_id   ON ratings (doc_id);
CREATE INDEX IF NOT EXISTS idx_ratings_rated_at ON ratings (rated_at);
CREATE INDEX IF NOT EXISTS idx_ratings_rating   ON ratings (rating);

-- ── Co-watch Pairs (PySpark output: 1.29M pairs) ──────────────
CREATE TABLE IF NOT EXISTS co_watch_pairs (
    pair_id         BIGSERIAL       PRIMARY KEY,
    doc_id_a        VARCHAR(64)     NOT NULL REFERENCES items(doc_id),
    doc_id_b        VARCHAR(64)     NOT NULL REFERENCES items(doc_id),
    co_watch_count  INTEGER         NOT NULL DEFAULT 0,  -- users who watched both
    co_watch_score  FLOAT           NOT NULL DEFAULT 0.0, -- normalised by popularity
    computed_at     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (doc_id_a, doc_id_b),
    CHECK (doc_id_a < doc_id_b)  -- avoid duplicate (A,B) and (B,A)
);

CREATE INDEX IF NOT EXISTS idx_cowatch_doc_a ON co_watch_pairs (doc_id_a);
CREATE INDEX IF NOT EXISTS idx_cowatch_doc_b ON co_watch_pairs (doc_id_b);
CREATE INDEX IF NOT EXISTS idx_cowatch_score ON co_watch_pairs (co_watch_score DESC);

-- ── Recommendations (search results served to users) ──────────
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id          BIGSERIAL       PRIMARY KEY,
    user_id         VARCHAR(64)     NOT NULL REFERENCES users(user_id),
    query           TEXT            NOT NULL,
    method          VARCHAR(32)     NOT NULL DEFAULT 'hybrid_ltr',  -- bm25|dense|hybrid|hybrid_ltr
    profile         VARCHAR(32)     NOT NULL DEFAULT 'chrisen',
    language        VARCHAR(64)     NOT NULL DEFAULT 'English',
    results         JSONB           NOT NULL,   -- [{doc_id, title, score, rank}]
    ndcg_at_10      FLOAT           DEFAULT NULL,
    latency_ms      INTEGER         DEFAULT NULL,
    cache_hit       BOOLEAN         NOT NULL DEFAULT FALSE,
    ltr_used        BOOLEAN         NOT NULL DEFAULT FALSE,
    served_at       TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recs_user_id   ON recommendations (user_id);
CREATE INDEX IF NOT EXISTS idx_recs_served_at ON recommendations (served_at);
CREATE INDEX IF NOT EXISTS idx_recs_method    ON recommendations (method);
CREATE INDEX IF NOT EXISTS idx_recs_cache_hit ON recommendations (cache_hit);

-- ── Impressions / Events (Kafka → propensity logger) ──────────
CREATE TABLE IF NOT EXISTS events (
    event_id        VARCHAR(128)    PRIMARY KEY,    -- UUID from Kafka
    user_id         VARCHAR(64)     NOT NULL REFERENCES users(user_id),
    doc_id          VARCHAR(64)     NOT NULL REFERENCES items(doc_id),
    event_type      VARCHAR(32)     NOT NULL,       -- click|watch_start|watch_complete|skip|dislike
    query           TEXT            DEFAULT NULL,
    position        SMALLINT        DEFAULT NULL,   -- rank position shown to user
    ltr_score       FLOAT           DEFAULT NULL,   -- raw LTR score at time of serving
    propensity      FLOAT           DEFAULT NULL,   -- P(shown | context) for IPW
    watch_pct       FLOAT           DEFAULT NULL CHECK (watch_pct IS NULL OR (watch_pct >= 0 AND watch_pct <= 1)),
    session_id      VARCHAR(128)    DEFAULT NULL,
    language        VARCHAR(64)     DEFAULT 'English',
    device_type     VARCHAR(32)     DEFAULT NULL,
    occurred_at     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_user_id     ON events (user_id);
CREATE INDEX IF NOT EXISTS idx_events_doc_id      ON events (doc_id);
CREATE INDEX IF NOT EXISTS idx_events_type        ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events (occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_session     ON events (session_id);

-- ── Explanations Cache (GPT-4o-mini, 7-day Redis TTL) ─────────
CREATE TABLE IF NOT EXISTS explanations (
    exp_id          BIGSERIAL       PRIMARY KEY,
    doc_id          VARCHAR(64)     NOT NULL REFERENCES items(doc_id),
    profile         VARCHAR(32)     NOT NULL,
    language        VARCHAR(64)     NOT NULL DEFAULT 'English',
    exp_type        VARCHAR(16)     NOT NULL DEFAULT 'why_this',  -- why_this|rag|vlm
    explanation     TEXT            NOT NULL,
    model           VARCHAR(64)     NOT NULL DEFAULT 'gpt-4o-mini',
    tokens_used     INTEGER         DEFAULT NULL,
    latency_ms      INTEGER         DEFAULT NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (doc_id, profile, language, exp_type)
);

CREATE INDEX IF NOT EXISTS idx_exp_doc_profile ON explanations (doc_id, profile, language);
CREATE INDEX IF NOT EXISTS idx_exp_created_at  ON explanations (created_at);

-- ── Model Artifacts (Metaflow versioning) ─────────────────────
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id     BIGSERIAL       PRIMARY KEY,
    run_id          VARCHAR(128)    NOT NULL UNIQUE,  -- Metaflow run ID
    flow_name       VARCHAR(128)    NOT NULL,
    artifact_type   VARCHAR(32)     NOT NULL,         -- bm25|dense|ltr|calibration
    s3_path         TEXT            NOT NULL,         -- MinIO S3 path
    ndcg_at_10      FLOAT           DEFAULT NULL,
    beir_ndcg       FLOAT           DEFAULT NULL,
    p99_latency_ms  INTEGER         DEFAULT NULL,
    all_gates_pass  BOOLEAN         NOT NULL DEFAULT FALSE,
    is_active       BOOLEAN         NOT NULL DEFAULT FALSE,
    promoted_at     TIMESTAMP       DEFAULT NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifacts_active ON model_artifacts (is_active);
CREATE INDEX IF NOT EXISTS idx_artifacts_flow   ON model_artifacts (flow_name);

-- ── Quality Gate Results ────────────────────────────────────── 
CREATE TABLE IF NOT EXISTS quality_gates (
    gate_id         BIGSERIAL       PRIMARY KEY,
    artifact_id     BIGINT          NOT NULL REFERENCES model_artifacts(artifact_id),
    gate_name       VARCHAR(64)     NOT NULL,   -- ltr_ndcg10|beir_ndcg|p99_cold_ms|...
    threshold       FLOAT           NOT NULL,
    measured_value  FLOAT           NOT NULL,
    passed          BOOLEAN         NOT NULL,
    evaluated_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_gates_artifact ON quality_gates (artifact_id);
CREATE INDEX IF NOT EXISTS idx_gates_passed   ON quality_gates (passed);

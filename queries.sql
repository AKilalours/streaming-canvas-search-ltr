-- ============================================================
-- StreamLens — Production SQL Queries
-- Akila Lourdes Miriyala Francis · MS in Artificial Intelligence
-- ============================================================
-- All queries run against schema.sql
-- Covers: SELECT, JOIN, GROUP BY, HAVING, window functions,
--         CTEs, subqueries, aggregation, ranking
-- ============================================================


-- ── Q1: Top 10 most-watched films overall ────────────────────
-- GROUP BY + ORDER BY + LIMIT
SELECT
    i.doc_id,
    i.title,
    i.genres,
    COUNT(e.event_id)                           AS total_watches,
    ROUND(AVG(e.watch_pct)::NUMERIC, 3)         AS avg_completion_rate,
    COUNT(DISTINCT e.user_id)                   AS unique_viewers
FROM items i
JOIN events e
    ON i.doc_id = e.doc_id
    AND e.event_type = 'watch_complete'
GROUP BY
    i.doc_id, i.title, i.genres
HAVING
    COUNT(DISTINCT e.user_id) >= 10             -- minimum audience threshold
ORDER BY
    total_watches DESC
LIMIT 10;


-- ── Q2: nDCG@10 by search method — ablation report ───────────
-- GROUP BY + AVG + CASE + HAVING
SELECT
    r.method,
    COUNT(r.rec_id)                             AS total_searches,
    ROUND(AVG(r.ndcg_at_10)::NUMERIC, 4)        AS avg_ndcg_at_10,
    ROUND(AVG(r.latency_ms)::NUMERIC, 1)        AS avg_latency_ms,
    ROUND(
        100.0 * SUM(CASE WHEN r.cache_hit THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0), 1
    )                                           AS cache_hit_rate_pct,
    ROUND(
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY r.latency_ms)::NUMERIC, 1
    )                                           AS p95_latency_ms,
    ROUND(
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY r.latency_ms)::NUMERIC, 1
    )                                           AS p99_latency_ms
FROM recommendations r
WHERE r.served_at >= NOW() - INTERVAL '7 days'
GROUP BY r.method
HAVING COUNT(*) >= 100
ORDER BY avg_ndcg_at_10 DESC;


-- ── Q3: User engagement funnel — click → watch → complete ─────
-- Multiple JOINs + GROUP BY + window function
WITH user_funnel AS (
    SELECT
        u.user_id,
        u.profile_type,
        u.language,
        COUNT(CASE WHEN e.event_type = 'click'          THEN 1 END) AS clicks,
        COUNT(CASE WHEN e.event_type = 'watch_start'    THEN 1 END) AS watch_starts,
        COUNT(CASE WHEN e.event_type = 'watch_complete' THEN 1 END) AS completions,
        COUNT(CASE WHEN e.event_type = 'skip'           THEN 1 END) AS skips,
        COUNT(CASE WHEN e.event_type = 'dislike'        THEN 1 END) AS dislikes,
        ROUND(AVG(e.watch_pct) FILTER (WHERE e.watch_pct IS NOT NULL)::NUMERIC, 3)
                                                                      AS avg_watch_pct
    FROM users u
    LEFT JOIN events e ON u.user_id = e.user_id
    WHERE e.occurred_at >= NOW() - INTERVAL '30 days'
    GROUP BY u.user_id, u.profile_type, u.language
)
SELECT
    profile_type,
    COUNT(DISTINCT user_id)                     AS active_users,
    SUM(clicks)                                 AS total_clicks,
    SUM(completions)                            AS total_completions,
    ROUND(
        100.0 * SUM(completions) / NULLIF(SUM(clicks), 0), 2
    )                                           AS click_to_complete_pct,
    ROUND(AVG(avg_watch_pct)::NUMERIC, 3)       AS avg_watch_completion,
    ROUND(
        100.0 * SUM(skips) / NULLIF(SUM(clicks), 0), 2
    )                                           AS skip_rate_pct
FROM user_funnel
GROUP BY profile_type
ORDER BY click_to_complete_pct DESC;


-- ── Q4: Co-watch graph — find films most similar to a given film
-- Self-JOIN on co_watch_pairs
SELECT
    i2.doc_id                                   AS similar_doc_id,
    i2.title                                    AS similar_title,
    i2.genres                                   AS similar_genres,
    cw.co_watch_count,
    ROUND(cw.co_watch_score::NUMERIC, 4)        AS co_watch_score
FROM co_watch_pairs cw
JOIN items i1
    ON (cw.doc_id_a = i1.doc_id OR cw.doc_id_b = i1.doc_id)
JOIN items i2
    ON (
        CASE WHEN cw.doc_id_a = i1.doc_id THEN cw.doc_id_b
             ELSE cw.doc_id_a END
    ) = i2.doc_id
WHERE
    i1.title ILIKE '%Pulp Fiction%'
    AND cw.co_watch_count >= 5
ORDER BY
    cw.co_watch_score DESC
LIMIT 15;


-- ── Q5: Genre popularity by user profile — recommendation pivot
-- GROUP BY + CROSS JOIN for genre extraction
SELECT
    u.profile_type,
    TRIM(genre_elem.genre)                      AS genre,
    COUNT(DISTINCT r.user_id)                   AS users_searched,
    COUNT(r.rec_id)                             AS total_recommendations,
    ROUND(AVG(r.ndcg_at_10)::NUMERIC, 4)        AS avg_ndcg
FROM recommendations r
JOIN users u ON r.user_id = u.user_id
CROSS JOIN LATERAL (
    SELECT UNNEST(STRING_TO_ARRAY(r.query, ' ')) AS genre
) genre_elem
WHERE
    r.served_at >= NOW() - INTERVAL '30 days'
    AND r.ndcg_at_10 IS NOT NULL
GROUP BY
    u.profile_type, TRIM(genre_elem.genre)
HAVING
    COUNT(r.rec_id) >= 20
ORDER BY
    u.profile_type, avg_ndcg DESC;


-- ── Q6: Model promotion history — quality gate audit ──────────
-- 3-table JOIN + GROUP BY + HAVING
SELECT
    ma.run_id,
    ma.flow_name,
    ma.artifact_type,
    ma.ndcg_at_10,
    ma.beir_ndcg,
    ma.p99_latency_ms,
    ma.all_gates_pass,
    ma.is_active,
    ma.promoted_at,
    COUNT(qg.gate_id)                           AS total_gates,
    SUM(CASE WHEN qg.passed THEN 1 ELSE 0 END)  AS gates_passed,
    SUM(CASE WHEN NOT qg.passed THEN 1 ELSE 0 END) AS gates_failed,
    STRING_AGG(
        CASE WHEN NOT qg.passed
             THEN qg.gate_name || '=' || ROUND(qg.measured_value::NUMERIC,4)
        END,
        ', '
    )                                           AS failed_gates
FROM model_artifacts ma
LEFT JOIN quality_gates qg ON ma.artifact_id = qg.artifact_id
WHERE ma.artifact_type = 'ltr'
GROUP BY
    ma.artifact_id, ma.run_id, ma.flow_name, ma.artifact_type,
    ma.ndcg_at_10, ma.beir_ndcg, ma.p99_latency_ms,
    ma.all_gates_pass, ma.is_active, ma.promoted_at
ORDER BY ma.created_at DESC
LIMIT 20;


-- ── Q7: Propensity-weighted uplift by position (IPW signal) ───
-- Inverse Probability Weighting — core causal inference query
-- Used by Doubly Robust IPW evaluator in StreamLens
WITH ipw_events AS (
    SELECT
        e.doc_id,
        e.position,
        e.watch_pct,
        e.propensity,
        -- IPW weight: 1/propensity for treated, 1/(1-propensity) for control
        CASE
            WHEN e.event_type = 'watch_complete'
            THEN (e.watch_pct / NULLIF(e.propensity, 0))
            ELSE 0.0
        END                                     AS ipw_reward
    FROM events e
    WHERE
        e.propensity IS NOT NULL
        AND e.propensity BETWEEN 0.05 AND 0.95  -- clip extreme weights
        AND e.occurred_at >= NOW() - INTERVAL '30 days'
)
SELECT
    position,
    COUNT(*)                                    AS n_impressions,
    ROUND(AVG(watch_pct) FILTER
        (WHERE watch_pct IS NOT NULL)::NUMERIC, 3) AS naive_completion,
    ROUND(AVG(ipw_reward)::NUMERIC, 4)          AS ipw_estimated_reward,
    ROUND(STDDEV(ipw_reward)::NUMERIC, 4)       AS reward_std,
    ROUND(AVG(propensity)::NUMERIC, 4)          AS avg_propensity
FROM ipw_events
GROUP BY position
HAVING COUNT(*) >= 50
ORDER BY position;


-- ── Q8: Real-time latency SLO monitoring — SRE dashboard query
-- Window functions + percentiles — feeds Grafana dashboard
SELECT
    DATE_TRUNC('hour', r.served_at)             AS hour_bucket,
    r.method,
    COUNT(*)                                    AS requests,
    ROUND(AVG(r.latency_ms)::NUMERIC, 1)        AS avg_ms,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
        (ORDER BY r.latency_ms)::NUMERIC, 1)    AS p50_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
        (ORDER BY r.latency_ms)::NUMERIC, 1)    AS p95_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP
        (ORDER BY r.latency_ms)::NUMERIC, 1)    AS p99_ms,
    -- SLO breach flag (target: p99 < 200ms)
    CASE
        WHEN PERCENTILE_CONT(0.99) WITHIN GROUP
            (ORDER BY r.latency_ms) > 200
        THEN 'SLO_BREACH'
        ELSE 'OK'
    END                                         AS slo_status,
    ROUND(
        100.0 * SUM(CASE WHEN r.cache_hit THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                           AS cache_hit_pct
FROM recommendations r
WHERE r.served_at >= NOW() - INTERVAL '24 hours'
GROUP BY
    DATE_TRUNC('hour', r.served_at), r.method
ORDER BY
    hour_bucket DESC, r.method;


-- ── Q9: Cold-start detection — users needing exploration boost ─
-- Subquery + EXISTS + GROUP BY
SELECT
    u.user_id,
    u.profile_type,
    u.language,
    u.interaction_count,
    u.cold_start,
    COALESCE(recent.genres_explored, 0)         AS genres_explored_7d,
    COALESCE(recent.watch_count_7d, 0)          AS watch_count_7d,
    -- Recommend Thompson Sampling ε boost for cold users
    CASE
        WHEN u.interaction_count < 5            THEN 'epsilon_0.40'
        WHEN u.interaction_count < 20           THEN 'epsilon_0.25'
        ELSE                                         'epsilon_0.15'
    END                                         AS recommended_epsilon
FROM users u
LEFT JOIN (
    SELECT
        e.user_id,
        COUNT(DISTINCT i.genres)                AS genres_explored,
        COUNT(CASE WHEN e.event_type = 'watch_complete' THEN 1 END) AS watch_count_7d
    FROM events e
    JOIN items i ON e.doc_id = i.doc_id
    WHERE e.occurred_at >= NOW() - INTERVAL '7 days'
    GROUP BY e.user_id
) recent ON u.user_id = recent.user_id
WHERE u.cold_start = TRUE
   OR u.interaction_count < 20
ORDER BY u.interaction_count ASC
LIMIT 100;


-- ── Q10: Explanation cost tracking — GenAI budget monitor ──────
-- GROUP BY + SUM + window function — feeds cost dashboard
SELECT
    DATE_TRUNC('day', ex.created_at)            AS day,
    ex.model,
    ex.exp_type,
    ex.language,
    COUNT(*)                                    AS explanations_generated,
    SUM(ex.tokens_used)                         AS total_tokens,
    -- GPT-4o-mini: ~$0.00015/1K input + $0.0006/1K output tokens
    ROUND(
        (SUM(ex.tokens_used) * 0.0008 / 1000.0)::NUMERIC, 4
    )                                           AS estimated_cost_usd,
    ROUND(AVG(ex.latency_ms)::NUMERIC, 1)       AS avg_latency_ms,
    -- Running total cost
    ROUND(SUM(SUM(ex.tokens_used) * 0.0008 / 1000.0)
        OVER (ORDER BY DATE_TRUNC('day', ex.created_at))::NUMERIC, 4
    )                                           AS cumulative_cost_usd
FROM explanations ex
WHERE ex.created_at >= NOW() - INTERVAL '30 days'
GROUP BY
    DATE_TRUNC('day', ex.created_at),
    ex.model, ex.exp_type, ex.language
ORDER BY
    day DESC, estimated_cost_usd DESC;

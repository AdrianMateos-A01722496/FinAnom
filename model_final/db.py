"""Persistence layer for FINANOM live demo.

Uses SQLite locally when DATABASE_URL is not set and SQLAlchemy-compatible URLs
for Azure SQL or other databases when it is set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from model_final import model as M
from model_final.scoring import CLEAN_COLUMNS, RUNTIME_COLUMNS, SCORE_COLUMNS, dashboard_record

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "output" / "finanom_demo.sqlite"
TABLE = "transacciones"
FEEDBACK_TABLE = "feedback_state"
LABELS_TABLE = "feedback_labels"

ALL_COLUMNS = RUNTIME_COLUMNS + CLEAN_COLUMNS + SCORE_COLUMNS


def database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    if url:
        return url
    DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB}"


def get_engine() -> Engine:
    return create_engine(database_url(), future=True)


def table_exists(engine: Engine, table: str = TABLE) -> bool:
    return inspect(engine).has_table(table)


def normalize_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(bool)
        elif "datetime" in str(out[col].dtype):
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif out[col].dtype == "object" or "string" in str(out[col].dtype):
            out[col] = out[col].where(out[col].notna(), None)
    return out


def seed_from_batch(state: dict | None = None) -> pd.DataFrame:
    """Build the recent-window table from existing clean data + scored batch report."""
    report = M.load_base_report()
    clean = pd.read_parquet(M.CLEAN_FILE, columns=CLEAN_COLUMNS)
    queued = M.compute_queue(report, state, clean_df=clean)
    ts = pd.to_datetime(queued["trace_t_timestamp"], errors="coerce")
    queued = queued.loc[ts >= pd.Timestamp("2025-03-11")].copy()
    row_ids = queued["trace_row_id"].astype(int).to_numpy()
    clean_rows = clean.iloc[row_ids].reset_index(drop=True)
    scored = queued[SCORE_COLUMNS].reset_index(drop=True)
    runtime = pd.DataFrame(
        {
            "tx_id": [f"batch-{int(i)}" for i in row_ids],
            "es_nueva": False,
            "estado": "Pendiente",
            "nota": "",
            "created_at": pd.Timestamp.now("UTC").isoformat(),
        }
    )
    return pd.concat([runtime, clean_rows, scored], axis=1)


def replace_transactions(engine: Engine, df: pd.DataFrame) -> int:
    data = normalize_for_sql(df[ALL_COLUMNS])
    data.to_sql(TABLE, engine, if_exists="replace", index=False, chunksize=5000)
    with engine.begin() as conn:
        try:
            conn.execute(text(f"CREATE INDEX ix_{TABLE}_tx_id ON {TABLE} (tx_id)"))
            conn.execute(text(f"CREATE INDEX ix_{TABLE}_is_anomaly ON {TABLE} (is_anomaly)"))
            conn.execute(text(f"CREATE INDEX ix_{TABLE}_severidad ON {TABLE} (severidad)"))
            conn.execute(text(f"CREATE INDEX ix_{TABLE}_timestamp ON {TABLE} (t_timestamp)"))
        except Exception:
            pass
    return int(len(data))


def ensure_store(engine: Engine | None = None) -> Engine:
    engine = engine or get_engine()
    if not table_exists(engine):
        replace_transactions(engine, seed_from_batch(M.load_state()))
    ensure_feedback_tables(engine)
    return engine


def ensure_feedback_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    with engine.begin() as conn:
        if not inspector.has_table(FEEDBACK_TABLE):
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {FEEDBACK_TABLE} (
                        id INTEGER PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            )
        if not inspector.has_table(LABELS_TABLE):
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {LABELS_TABLE} (
                        trace_row_id TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        nota TEXT,
                        revisor TEXT,
                        timestamp_revision TEXT NOT NULL
                    )
                    """
                )
            )


def load_feedback_state(engine: Engine) -> dict | None:
    ensure_feedback_tables(engine)
    with engine.begin() as conn:
        row = conn.execute(
            text(f"SELECT state_json FROM {FEEDBACK_TABLE} WHERE id = 1")
        ).fetchone()
    if row is None:
        return M.load_state()
    import json

    return json.loads(row[0])


def save_feedback_state(engine: Engine, state: dict) -> None:
    ensure_feedback_tables(engine)
    import json

    payload = json.dumps(state, ensure_ascii=False)
    now = pd.Timestamp.now("UTC").isoformat()
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {FEEDBACK_TABLE} WHERE id = 1"))
        conn.execute(
            text(
                f"INSERT INTO {FEEDBACK_TABLE} (id, state_json, updated_at) "
                "VALUES (1, :state_json, :updated_at)"
            ),
            {"state_json": payload, "updated_at": now},
        )


def append_feedback_labels(engine: Engine, labels: pd.DataFrame, revisor: str) -> None:
    ensure_feedback_tables(engine)
    if labels.empty:
        return
    out = labels.copy()
    out["trace_row_id"] = out["trace_row_id"].astype(str)
    out["nota"] = out.get("nota", "").fillna("")
    out["revisor"] = revisor
    out["timestamp_revision"] = pd.Timestamp.now("UTC").isoformat()
    out[["trace_row_id", "decision", "nota", "revisor", "timestamp_revision"]].to_sql(
        LABELS_TABLE, engine, if_exists="append", index=False
    )


def load_window(engine: Engine) -> pd.DataFrame:
    ensure_store(engine)
    return pd.read_sql(f"SELECT * FROM {TABLE}", engine)


def save_scored_window(engine: Engine, df: pd.DataFrame) -> int:
    return replace_transactions(engine, df)


def update_review_states(engine: Engine, reviews: pd.DataFrame) -> None:
    if reviews.empty:
        return
    with engine.begin() as conn:
        for row in reviews.to_dict(orient="records"):
            conn.execute(
                text(
                    f"UPDATE {TABLE} SET estado = :estado, nota = :nota "
                    "WHERE tx_id = :tx_id"
                ),
                {
                    "estado": row.get("estado") or "Pendiente",
                    "nota": row.get("nota") or "",
                    "tx_id": str(row["trace_row_id"]),
                },
            )


def _severity_where(value: str) -> tuple[str, dict[str, Any]]:
    if not value:
        return "", {}
    mapping = {
        "Alta": ["CRITICO"],
        "Media": ["ALTO"],
        "Informativa": ["MEDIO", "BAJO"],
        "CRITICO": ["CRITICO"],
        "ALTO": ["ALTO"],
        "MEDIO": ["MEDIO"],
        "BAJO": ["BAJO"],
    }
    values = mapping.get(value, [value])
    params = {f"sev_{i}": v for i, v in enumerate(values)}
    names = ", ".join(f":{k}" for k in params)
    return f"severidad IN ({names})", params


def query_transactions(
    engine: Engine,
    filter_mode: str = "anomalias",
    severidad: str = "",
    q: str = "",
    page: int = 1,
    page_size: int = 25,
) -> dict[str, Any]:
    ensure_store(engine)
    page = max(1, int(page or 1))
    page_size = min(200, max(1, int(page_size or 25)))

    clauses: list[str] = []
    params: dict[str, Any] = {}
    if filter_mode == "anomalias":
        clauses.append("is_anomaly = 1")
    sev_clause, sev_params = _severity_where(severidad)
    if sev_clause:
        clauses.append(sev_clause)
        params.update(sev_params)
    if q:
        clauses.append(
            "(CAST(t_folio AS TEXT) LIKE :q OR CAST(t_cuarto AS TEXT) LIKE :q "
            "OR CAST(t_codigo AS TEXT) LIKE :q OR CAST(t_referencia AS TEXT) LIKE :q)"
        )
        params["q"] = f"%{q}%"
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    offset = (page - 1) * page_size

    with engine.begin() as conn:
        total = int(conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}{where}"), params).scalar() or 0)
        total_anom = int(conn.execute(text(f"SELECT COUNT(*) FROM {TABLE} WHERE is_anomaly = 1")).scalar() or 0)
        total_new = int(conn.execute(text(f"SELECT COUNT(*) FROM {TABLE} WHERE es_nueva = 1")).scalar() or 0)
        if engine.dialect.name.startswith("mssql"):
            sql = (
                f"SELECT * FROM {TABLE}{where} "
                "ORDER BY es_nueva DESC, is_anomaly DESC, anomaly_rank ASC, t_timestamp DESC "
                "OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY"
            )
        else:
            sql = (
                f"SELECT * FROM {TABLE}{where} "
                "ORDER BY es_nueva DESC, is_anomaly DESC, anomaly_rank ASC, t_timestamp DESC "
                "LIMIT :limit OFFSET :offset"
            )
        rows = conn.execute(text(sql), {**params, "limit": page_size, "offset": offset}).mappings().all()

    records = [dashboard_record(pd.Series(dict(row))) for row in rows]
    total_pages = max(1, (total + page_size - 1) // page_size)
    return {
        "records": records,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "summary": {"total_anomalias": total_anom, "total_nuevas": total_new},
    }

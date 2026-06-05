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
NEW_BADGE_TTL_MINUTES = 10
CATEGORY_FILTERS = {
    "Monto atipico": "MONTO_ATIPICO",
    "Cancelacion irregular": "CANCELACION",
    "Fuera de estancia": "FUERA_DE_ESTANCIA",
    "Inconsistencia de reservacion": "RESERVACION",
    "Inconsistencia de signo": "INCONSISTENCIA_DE_SIGNO",
    "Metodo de pago": "METODO_PAGO",
    "Posible duplicado": "DUPLICADO",
    "Patron atipico (modelo)": "ATIPICO_IF",
}


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


def expire_new_flags(engine: Engine) -> None:
    cutoff = (pd.Timestamp.now("UTC") - pd.Timedelta(minutes=NEW_BADGE_TTL_MINUTES)).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {TABLE} SET es_nueva = 0 WHERE es_nueva = 1 AND created_at < :cutoff"),
            {"cutoff": cutoff},
        )


def latest_day_range(engine: Engine) -> tuple[str | None, str | None]:
    ensure_store(engine)
    with engine.begin() as conn:
        value = conn.execute(text(f"SELECT MAX(t_timestamp) FROM {TABLE}")).scalar()
    latest = pd.to_datetime(value, errors="coerce", format="mixed")
    if pd.isna(latest):
        return None, None
    start = latest.normalize()
    end = start + pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


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


def _text_expr(engine: Engine, column: str) -> str:
    if engine.dialect.name.startswith("mssql"):
        return f"CAST({column} AS NVARCHAR(MAX))"
    return f"CAST({column} AS TEXT)"


def query_transactions(
    engine: Engine,
    filter_mode: str = "anomalias",
    severidad: str = "",
    q: str = "",
    scope: str = "all",
    date_from: str = "",
    date_to: str = "",
    folio: str = "",
    cuarto: str = "",
    concepto: str = "",
    categoria: str = "",
    monto_min: str = "",
    monto_max: str = "",
    sort_by: str = "priority",
    sort_dir: str = "asc",
    page: int = 1,
    page_size: int = 25,
) -> dict[str, Any]:
    ensure_store(engine)
    expire_new_flags(engine)
    page = max(1, int(page or 1))
    page_size = min(10000, max(1, int(page_size or 25)))

    clauses: list[str] = []
    params: dict[str, Any] = {}
    latest_start, latest_end = latest_day_range(engine)
    selected_day = latest_start[:10] if latest_start else None
    if scope == "today":
        requested_day = pd.to_datetime(date_from, errors="coerce") if date_from else pd.NaT
        if pd.notna(requested_day):
            selected_day = requested_day.strftime("%Y-%m-%d")
            params["selected_start"] = f"{selected_day} 00:00:00"
            params["selected_end"] = (requested_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
            clauses.append("t_timestamp >= :selected_start AND t_timestamp < :selected_end")
        elif latest_start and latest_end:
            clauses.append("t_timestamp >= :latest_start AND t_timestamp < :latest_end")
            params.update({"latest_start": latest_start, "latest_end": latest_end})
    elif date_from:
        clauses.append("t_timestamp >= :date_from")
        params["date_from"] = f"{date_from} 00:00:00"
    if scope != "today" and date_to:
        clauses.append("t_timestamp < :date_to")
        end = pd.to_datetime(date_to, errors="coerce") + pd.Timedelta(days=1)
        params["date_to"] = end.strftime("%Y-%m-%d %H:%M:%S")
    if filter_mode == "anomalias":
        clauses.append("is_anomaly = 1")
    sev_clause, sev_params = _severity_where(severidad)
    if sev_clause:
        clauses.append(sev_clause)
        params.update(sev_params)
    if q:
        clauses.append(
            f"({_text_expr(engine, 't_folio')} LIKE :q OR {_text_expr(engine, 't_cuarto')} LIKE :q "
            f"OR {_text_expr(engine, 't_codigo')} LIKE :q OR {_text_expr(engine, 't_referencia')} LIKE :q)"
        )
        params["q"] = f"%{q}%"
    if folio:
        clauses.append(f"{_text_expr(engine, 't_folio')} LIKE :folio")
        params["folio"] = f"%{folio}%"
    if cuarto:
        clauses.append(f"{_text_expr(engine, 't_cuarto')} LIKE :cuarto")
        params["cuarto"] = f"%{cuarto}%"
    if concepto:
        clauses.append(f"{_text_expr(engine, 't_codigo')} LIKE :concepto")
        params["concepto"] = f"%{concepto}%"
    if categoria:
        if categoria == "Transaccion normal":
            clauses.append("(tipo_inconsistencia IS NULL OR tipo_inconsistencia = '')")
        else:
            clauses.append(f"{_text_expr(engine, 'tipo_inconsistencia')} LIKE :categoria")
            params["categoria"] = f"%{CATEGORY_FILTERS.get(categoria, categoria.upper().replace(' ', '_'))}%"
    if monto_min:
        clauses.append("t_monto >= :monto_min")
        params["monto_min"] = float(monto_min)
    if monto_max:
        clauses.append("t_monto <= :monto_max")
        params["monto_max"] = float(monto_max)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    offset = (page - 1) * page_size
    priority_rank = "daily_rank" if scope == "today" else "anomaly_rank"
    sort_map = {
        "priority": f"es_nueva DESC, is_anomaly DESC, {priority_rank} ASC, t_timestamp DESC",
        "fecha": "t_timestamp",
        "hora": "t_timestamp",
        "monto": "t_monto",
        "folio": "t_folio",
        "concepto": "t_codigo",
        "severidad": "severidad",
    }
    if sort_by not in sort_map:
        sort_by = "priority"
    if sort_by == "priority":
        order_by = sort_map[sort_by]
    else:
        direction = "DESC" if str(sort_dir).lower() == "desc" else "ASC"
        order_by = f"es_nueva DESC, {sort_map[sort_by]} {direction}"

    with engine.begin() as conn:
        total_global = int(conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar() or 0)
        total = int(conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}{where}"), params).scalar() or 0)
        total_anom = int(conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLE}{where} AND is_anomaly = 1" if where else f"SELECT COUNT(*) FROM {TABLE} WHERE is_anomaly = 1"),
            params,
        ).scalar() or 0)
        total_new = int(conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLE}{where} AND es_nueva = 1" if where else f"SELECT COUNT(*) FROM {TABLE} WHERE es_nueva = 1"),
            params,
        ).scalar() or 0)
        if scope == "today":
            if "selected_start" in params:
                base_where = " WHERE t_timestamp >= :selected_start AND t_timestamp < :selected_end"
            elif "latest_start" in params:
                base_where = " WHERE t_timestamp >= :latest_start AND t_timestamp < :latest_end"
            else:
                base_where = ""
            from_expr = (
                "(SELECT *, ROW_NUMBER() OVER ("
                "ORDER BY CASE severidad WHEN 'CRITICO' THEN 0 WHEN 'ALTO' THEN 1 WHEN 'MEDIO' THEN 2 ELSE 3 END, "
                "anomaly_score DESC, anomaly_rank ASC, t_timestamp DESC"
                f") AS daily_rank FROM {TABLE}{base_where}) ranked"
            )
        else:
            from_expr = TABLE
        if engine.dialect.name.startswith("mssql"):
            sql = (
                f"SELECT * FROM {from_expr}{where} "
                f"ORDER BY {order_by} "
                "OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY"
            )
        else:
            sql = (
                f"SELECT * FROM {from_expr}{where} "
                f"ORDER BY {order_by} "
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
        "summary": {
            "total_global": total_global,
            "total_anomalias": total_anom,
            "total_nuevas": total_new,
            "latest_day": latest_start[:10] if latest_start else None,
            "selected_day": selected_day,
            "new_badge_ttl_minutes": NEW_BADGE_TTL_MINUTES,
        },
    }

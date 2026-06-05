"""Seed FINANOM transaction storage from local parquet artifacts.

Local usage:
    uv run python model_final/migrate_to_sql.py

Azure SQL usage:
    DATABASE_URL='mssql+pyodbc://...' uv run python model_final/migrate_to_sql.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_final import db  # noqa: E402
from model_final import model as M  # noqa: E402


def migrate(force: bool = False) -> None:
    engine = db.get_engine()
    if db.table_exists(engine) and not force:
        print(f"Tabla {db.TABLE!r} ya existe. Usa --force para reemplazarla.")
        return
    rows = db.seed_from_batch(M.load_state())
    written = db.replace_transactions(engine, rows)
    db.ensure_feedback_tables(engine)
    print(f"Migracion completa: {written:,} transacciones en {db.TABLE!r}.")
    print(f"Backend SQLAlchemy: {engine.url.render_as_string(hide_password=True)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Reemplaza la tabla si ya existe.")
    args = parser.parse_args()
    migrate(force=args.force)


if __name__ == "__main__":
    main()

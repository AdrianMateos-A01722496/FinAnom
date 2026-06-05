#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?Define DATABASE_URL antes de migrar.}"

uv run python model_final/migrate_to_sql.py "${@}"

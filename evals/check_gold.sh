#!/usr/bin/env bash
# Runs every gold query in a questions file against the local database.
# Usage (from the project root): ./check_gold.sh [questions.draft.jsonl]
set -euo pipefail
FILE="${1:-questions.draft.jsonl}"
jq -r 'select((.gold_sql // .sql) != null) | [.id, .question, (.gold_sql // .sql)] | @tsv' "$FILE" |
while IFS=$'\t' read -r id question sql; do
  echo "=================================================================="
  echo "$id  $question"
  echo "------------------------------------------------------------------"
  docker compose exec -T db psql -U finquery -d finquery -P pager=off -c "$sql" < /dev/null || echo "!! $id FAILED"
done
echo "=================================================================="
echo "Out-of-scope questions (no SQL, must be refused):"
jq -r 'select((.gold_sql // .sql) == null) | "  \(.id)  \(.question)"' "$FILE"

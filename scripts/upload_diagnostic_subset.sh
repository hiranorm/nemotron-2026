#!/bin/bash
# 診断用の stratified subset CSV を Kaggle Private Dataset としてアップロード/更新する。
#
# Usage:
#   ./scripts/upload_diagnostic_subset.sh           # 新規作成
#   ./scripts/upload_diagnostic_subset.sh --update  # 既存を更新
#
# 参照パス（Kaggle 上）:
#   /kaggle/input/datasets/hiranorm/nemotron-diagnostic-subset/diagnostic_subset.csv

set -e

UPDATE_FLAG="$1"
KAGGLE="$HOME/kaggle_workspace/.venv/bin/kaggle"
KAGGLE_USER="hiranorm"
DATASET_SLUG="nemotron-diagnostic-subset"
DATASET_TITLE="Nemotron Diagnostic Subset"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
STAGE_DIR="/tmp/kaggle_upload_${DATASET_SLUG}"

SRC_CSV="$PROJECT_DIR/data/inputs/diagnostic_subset.csv"
if [ ! -f "$SRC_CSV" ]; then
    echo "ERROR: $SRC_CSV がありません。先に scripts/categorize_train.py を実行してください。"
    exit 1
fi

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp "$SRC_CSV" "$STAGE_DIR/"

cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "id": "${KAGGLE_USER}/${DATASET_SLUG}",
  "title": "${DATASET_TITLE}",
  "licenses": [{"name": "unknown"}]
}
EOF

echo "ステージング内容:"
find "$STAGE_DIR" -type f | sort | while read f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  $size  ${f#$STAGE_DIR/}"
done

if [ "$UPDATE_FLAG" = "--update" ]; then
    $KAGGLE datasets version \
        -p "$STAGE_DIR" \
        -m "diagnostic subset refreshed ($(date '+%Y-%m-%d %H:%M'))" \
        --dir-mode zip
else
    $KAGGLE datasets create \
        -p "$STAGE_DIR" \
        --dir-mode zip
    echo "※ 2回目以降は --update を付けてください"
fi

rm -rf "$STAGE_DIR"
echo "完了: https://www.kaggle.com/datasets/${KAGGLE_USER}/${DATASET_SLUG}"

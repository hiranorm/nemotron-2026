#!/bin/bash
# EXP コードを Kaggle Dataset にアップロードするスクリプト
#
# Usage:
#   ./scripts/upload_code_to_kaggle.sh           # 初回作成
#   ./scripts/upload_code_to_kaggle.sh --update  # バージョン更新（2回目以降）
#
# アップロード内容:
#   - EXP/EXP000/train.py
#   - EXP/EXP000/infer.py
#   - EXP/EXP000/config/*.yaml
#
# Kaggle Notebooks での参照パス:
#   /kaggle/input/nemotron-reasoning-exp-code/EXP/EXP000/
#
# 前提: Kaggle API 認証済み（~/.kaggle/kaggle.json）

set -e

KAGGLE="$HOME/.local/bin/kaggle"
KAGGLE_USER="hiranorm"
DATASET_SLUG="nemotron-reasoning-exp-code"
DATASET_TITLE="Nemotron Reasoning EXP Code"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
STAGE_DIR="/tmp/kaggle_upload_${DATASET_SLUG}"

echo "=== Kaggle Dataset アップロード: ${KAGGLE_USER}/${DATASET_SLUG} ==="
echo ""

# ステージングディレクトリ準備
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/EXP/EXP000/config"
mkdir -p "$STAGE_DIR/EXP/EXP001/config"

copy_exp() {
    local exp="$1"
    local src="$PROJECT_DIR/EXP/$exp"
    local dst="$STAGE_DIR/EXP/$exp"
    echo "--- $exp ---"
    for f in train.py infer.py; do
        if [ -f "$src/$f" ]; then
            cp "$src/$f" "$dst/"
            echo "  $f"
        else
            echo "  WARNING: $f が見つかりません（スキップ）"
        fi
    done
    local yaml_count
    yaml_count=$(ls "$src/config/"*.yaml 2>/dev/null | wc -l | tr -d ' ')
    if [ "$yaml_count" -eq 0 ]; then
        echo "  WARNING: config/*.yaml が見つかりません"
    else
        cp "$src/config/"*.yaml "$dst/config/"
        for f in "$src/config/"*.yaml; do
            echo "  config/$(basename "$f")"
        done
    fi
}

# EXP コードをコピー
echo "[1/2] EXP コードをコピー中..."
copy_exp "EXP000"
copy_exp "EXP001"

# dataset-metadata.json を生成
echo "[2/2] メタデータ生成中..."
cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "id": "${KAGGLE_USER}/${DATASET_SLUG}",
  "title": "${DATASET_TITLE}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo ""
echo "ステージング内容:"
find "$STAGE_DIR" -type f | sort | while read f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  $size  ${f#$STAGE_DIR/}"
done
echo ""

# アップロード
if [ "$1" = "--update" ]; then
    echo "バージョン更新中..."
    $KAGGLE datasets version \
        -p "$STAGE_DIR" \
        -m "EXP001 CoT-SFT added ($(date '+%Y-%m-%d %H:%M'))" \
        --dir-mode zip
    echo ""
    echo "完了: https://www.kaggle.com/datasets/${KAGGLE_USER}/${DATASET_SLUG}"
else
    echo "Dataset 新規作成中..."
    $KAGGLE datasets create \
        -p "$STAGE_DIR" \
        --dir-mode zip
    echo ""
    echo "完了: https://www.kaggle.com/datasets/${KAGGLE_USER}/${DATASET_SLUG}"
    echo ""
    echo "※ 2回目以降は --update オプションを使用してください"
    echo ""
    echo "submit.ipynb の EXP_CODE_DATASET を以下に設定してください:"
    echo "  EXP_CODE_DATASET = \"${DATASET_SLUG}\""
fi

# クリーンアップ
rm -rf "$STAGE_DIR"
echo "ステージングディレクトリを削除しました"

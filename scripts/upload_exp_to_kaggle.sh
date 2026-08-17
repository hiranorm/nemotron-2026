#!/bin/bash
# EXP の config を Kaggle Dataset にアップロードし、実行用 run notebook を生成するスクリプト
#
# Usage:
#   ./scripts/upload_exp_to_kaggle.sh EXP002 child-exp001           # Dataset 作成 + notebook 生成
#   ./scripts/upload_exp_to_kaggle.sh EXP002 child-exp001 --update  # Dataset 更新 + notebook 再生成
#
# アップロード内容（Dataset）:
#   - EXP/{EXP}*/config/{CHILD_EXP}.yaml
#
# ローカル生成（Kaggle Kernel としてアップロードするもの）:
#   - notebooks/run-{EXP}-{CHILD_EXP}.ipynb
#     ← EXP/{EXP}*/train.ipynb から CHILD_EXP を固定して生成
#
# Kaggle での参照パス:
#   /kaggle/input/{user}/nemotron-{exp}-{child-exp}/{child-exp}.yaml
#
# 前提: Kaggle API 認証済み（~/.kaggle/kaggle.json）

set -e

# ── 引数チェック ────────────────────────────────────────────────────────────
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 EXP002 child-exp001 [--update]"
    echo "  EXP:       例) EXP002"
    echo "  CHILD_EXP: 例) child-exp001"
    exit 1
fi

EXP="$1"         # 例: EXP002
CHILD_EXP="$2"   # 例: child-exp001
UPDATE_FLAG="$3"  # --update または空

# ── 変数設定 ────────────────────────────────────────────────────────────────
KAGGLE="$HOME/kaggle_workspace/.venv/bin/kaggle"
KAGGLE_USER="hiranorm"

EXP_LOWER=$(echo "$EXP" | tr '[:upper:]' '[:lower:]')            # exp002
DATASET_SLUG="nemotron-${EXP_LOWER}-${CHILD_EXP}"               # nemotron-exp002-child-exp001
DATASET_TITLE="Nemotron ${EXP} ${CHILD_EXP} Config"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
STAGE_DIR="/tmp/kaggle_upload_${DATASET_SLUG}"

echo "=== Kaggle Dataset アップロード: ${KAGGLE_USER}/${DATASET_SLUG} ==="
echo ""

# ── EXP ディレクトリを探す ───────────────────────────────────────────────────
EXP_DIR=$(find "$PROJECT_DIR/EXP" -maxdepth 1 -type d -name "${EXP}*" | sort | head -1)
if [ -z "$EXP_DIR" ]; then
    echo "ERROR: EXP ディレクトリが見つかりません: $PROJECT_DIR/EXP/${EXP}*"
    exit 1
fi
echo "EXP dir: $EXP_DIR"

# ── config yaml を確認 ──────────────────────────────────────────────────────
CONFIG_FILE="$EXP_DIR/config/${CHILD_EXP}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config が見つかりません: $CONFIG_FILE"
    exit 1
fi
echo "Config:  $CONFIG_FILE"
echo ""

# ── ステージングディレクトリ準備 ────────────────────────────────────────────
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# [1/3] config yaml をコピー（Kaggle 参照パス: /kaggle/input/{user}/{slug}/{child-exp}.yaml）
echo "[1/3] config をコピー中..."
cp "$CONFIG_FILE" "$STAGE_DIR/"
echo "  ${CHILD_EXP}.yaml"

# [2/3] run notebook を生成
echo "[2/3] run notebook を生成中..."
EXP="$EXP" CHILD_EXP="$CHILD_EXP" DATASET_SLUG="$DATASET_SLUG" KAGGLE_USER="$KAGGLE_USER" \
EXP_DIR="$EXP_DIR" PROJECT_DIR="$PROJECT_DIR" \
python3 << 'PYEOF'
import os, json, pathlib, re

exp         = os.environ["EXP"]
child_exp   = os.environ["CHILD_EXP"]
slug        = os.environ["DATASET_SLUG"]
kaggle_user = os.environ["KAGGLE_USER"]
exp_dir     = pathlib.Path(os.environ["EXP_DIR"])
project_dir = pathlib.Path(os.environ["PROJECT_DIR"])

train_nb_path = exp_dir / "train.ipynb"
if not train_nb_path.exists():
    raise FileNotFoundError(f"train.ipynb not found: {train_nb_path}")

nb = json.loads(train_nb_path.read_text())

# CHILD_EXP セルを書き換え（cell[0]）
patched = False
for cell in nb["cells"]:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if 'CHILD_EXP = ' in src and '_cfg_path' in src:
        # CHILD_EXP 値を固定
        src = re.sub(
            r'CHILD_EXP = "[^"]*".*',
            f'CHILD_EXP = "{child_exp}"  # FIXED — Kaggle 実行用',
            src,
            count=1,
        )
        # config のパスを Dataset path に変更
        # この Kaggle 環境のマウント規約: /kaggle/input/datasets/{owner}/{slug}/{child_exp}.yaml
        # （train.ipynb 側に glob フォールバックがあるので多少ズレても自動発見する）
        src = src.replace(
            '_cfg_path = pathlib.Path(f"config/{CHILD_EXP}.yaml")',
            f'_cfg_path = pathlib.Path(f"/kaggle/input/datasets/{kaggle_user}/{slug}/{{CHILD_EXP}}.yaml")',
        )
        cell["source"] = src
        patched = True
        break

if not patched:
    print("WARNING: CHILD_EXP セルが見つかりませんでした（パッチをスキップ）")

# notebooks/ に書き出す
notebooks_dir = project_dir / "notebooks"
notebooks_dir.mkdir(exist_ok=True)
out_path = notebooks_dir / f"run-{exp}-{child_exp}.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"  notebooks/run-{exp}-{child_exp}.ipynb")
PYEOF

# [3/3] dataset-metadata.json を生成
echo "[3/3] メタデータ生成中..."
cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "id": "${KAGGLE_USER}/${DATASET_SLUG}",
  "title": "${DATASET_TITLE}",
  "licenses": [{"name": "unknown"}]
}
EOF

echo ""
echo "ステージング内容:"
find "$STAGE_DIR" -type f | sort | while read f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  $size  ${f#$STAGE_DIR/}"
done
echo ""

# ── アップロード ──────────────────────────────────────────────────────────────
if [ "$UPDATE_FLAG" = "--update" ]; then
    echo "バージョン更新中..."
    $KAGGLE datasets version \
        -p "$STAGE_DIR" \
        -m "${EXP} ${CHILD_EXP} config updated ($(date '+%Y-%m-%d %H:%M'))" \
        --dir-mode zip
else
    echo "Dataset 新規作成中..."
    $KAGGLE datasets create \
        -p "$STAGE_DIR" \
        --dir-mode zip
    echo ""
    echo "※ 2回目以降は --update オプションを使用してください"
fi

echo ""
echo "完了: https://www.kaggle.com/datasets/${KAGGLE_USER}/${DATASET_SLUG}"
echo ""
echo "次のステップ:"
echo "  1. Kaggle Notebook を新規作成"
echo "  2. 上記 Dataset をアタッチ"
echo "  3. notebooks/run-${EXP}-${CHILD_EXP}.ipynb の内容をコピーして実行"
echo "     または kaggle kernels push で直接アップロード"

# ── クリーンアップ ────────────────────────────────────────────────────────────
rm -rf "$STAGE_DIR"
echo "ステージングディレクトリを削除しました"

# Outfit AI

ファッションアイテムを管理・推奨するFlaskアプリケーション。

## セットアップ

```bash
python -m venv .venv
.\.venv\Scripts\Activate.bat
pip install -r requirements.txt
python run.py
```

## ディレクトリ構成

- `run.py`: アプリ起動の入口
- `requirements.txt`: 必要ライブラリ一覧
- `.env`: APIキーなど秘密情報
- `app/`: Flaskアプリ本体
  - `routes/`: URLと処理
  - `templates/`: HTML
  - `static/`: CSS・画像
  - `models/`: DB関連
  - `services/`: AI・ビジネスロジック

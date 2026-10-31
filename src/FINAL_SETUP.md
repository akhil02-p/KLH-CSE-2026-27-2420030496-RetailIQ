# RetailIQ V1.0 — Final Frontend Setup

This package is a **complete RetailIQ project**, not a frontend-only folder.

## Structure

- `app.py` — premium Streamlit frontend
- `src/` — existing RetailIQ analytics/ML engines
- `assets/retailiq_logo.svg` — new branding
- `.streamlit/config.toml` — Streamlit configuration
- `requirements.txt` — dependencies
- `RUN_RETAILIQ.bat` — Windows launcher

## Run in VS Code

Open this exact folder as the VS Code workspace:

`RetailIQ_v08_work`

Then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Important

Do **not** copy only `app.py` into a separate empty folder. The frontend imports the existing backend through `src.*`.

The application must be launched from the project root containing both `app.py` and `src/`.

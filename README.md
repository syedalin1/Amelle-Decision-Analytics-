# Amelle Decision Analytics — Version 1

A simple teaching-oriented business analytics app.

## Five-stage workflow

1. **Understand My Data** — diagnoses the uploaded dataset without changing it.
2. **Clean My Data** — lets the user choose cleaning actions and download cleaned data.
3. **Visualise My Data** — recommends a simple chart and lets the user approve/change it.
4. **Analyse My Data** — recommends a simple analysis based on selected variables; the user must approve it.
5. **AI Decision Support** — interprets approved analytical evidence as finding, interpretation, possible actions and limitations.

## Important V1 boundaries

- CSV and Excel are supported.
- Excel V1 reads the first worksheet.
- No automatic predictive machine learning.
- The app does not promise to find every possible data-quality issue.
- Potential outliers are flagged for review, not automatically removed.
- Do not upload personal, sensitive, confidential, or commercially restricted data.
- The human user remains responsible for the final business decision.

## Run locally

Install the requirements:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## AI layer

Levels 1–4 work without a Gemini API key.

For Level 5, configure `GEMINI_API_KEY` as an environment variable or in the
project's `.streamlit/secrets.toml` file. This secrets file is excluded by
`.gitignore` and must not be committed.

Example Streamlit secret:

```toml
GEMINI_API_KEY = "your-key-here"
```

Do **not** put an API key directly inside `app.py` and do not commit keys to GitHub.

## Suggested first tests

1. Test with the existing coffee-shop teaching dataset.
2. Test with a completely different anonymised dataset (for example HR or marketing).
3. Note anything that only works for one dataset — that identifies what still needs to become more generic.

# Mansik Santulan Score — local running website

A FastAPI + scikit-learn website that uses the supplied trained model to estimate a student wellbeing score from lifestyle and social-media inputs.

## Project structure

- `main.py` — FastAPI backend and model prediction endpoint
- `Mental_Health_Model.pkl` — supplied trained model
- `Student Social Media And Mental Health Impact.csv` — supplied dataset
- `frontend/index.html` — website page
- `frontend/style.css` — responsive styling
- `frontend/script.js` — form validation, API call, score display
- `requirements.txt` — Python dependencies

## Run on Windows

1. Install Python 3.11 or 3.12 (recommended for package compatibility).
2. Extract this ZIP to a folder.
3. Open Command Prompt/Terminal in that folder.
4. Create and activate a virtual environment:

   ```bat
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the website:

   ```bat
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

7. Open `http://127.0.0.1:8000` in your browser.

API docs: `http://127.0.0.1:8000/docs`
Health check: `http://127.0.0.1:8000/api/health`

## Notes

- The website uses the API on the same origin; no separate frontend server or CORS configuration is needed.
- Keep `Mental_Health_Model.pkl` beside `main.py`.
- The saved model was created with scikit-learn 1.9.0. Loading pickle/joblib models with a different scikit-learn version can cause warnings or compatibility problems; use the pinned dependency above where available.
- The score is a machine-learning estimate based on the supplied dataset. It is not a mental-health diagnosis, and should not be used to make clinical decisions.
- For public deployment, configure allowed origins, HTTPS, and appropriate privacy/security controls instead of exposing development settings.


## Troubleshooting

- Keep the terminal running while using the site.
- Test backend health at `http://127.0.0.1:8000/api/health`.
- If the page looks cut off, press `Ctrl + F5` in the browser to force-refresh the updated CSS.
- If prediction fails, read the terminal traceback; it often indicates a model feature-name or package-version mismatch.

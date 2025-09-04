
# MO-CCI Safety Prediction – MVP

This repo contains:
- `model_event.pkl` – Event type prediction pipeline (TF-IDF + Logistic Regression)
- `model_severity.pkl` – Severity prediction pipeline (TF-IDF + Logistic Regression; Hospitalization vs Recordable)
- `hazard_checklist.csv` – Mapping from hazard buckets to preventive controls
- `app_mocci_safety.py` – Streamlit app (planning-level inputs → predictions → checklist)

## How to run locally

1) Install Python 3.9+ and run:
   ```bash
   pip install streamlit scikit-learn pandas numpy
   ```

2) Put these four files in the same folder:
   - model_event.pkl
   - model_severity.pkl
   - hazard_checklist.csv
   - app_mocci_safety.py

3) Start the app:
   ```bash
   streamlit run app_mocci_safety.py
   ```

4) Open the URL shown by Streamlit (usually http://localhost:8501).

## Notes
- Models were trained on OSHA SIR construction incidents (2015–2025) with NAICS+STATE tokens and narrative text.
- The app maps your high-level inputs to representative tokens and exposure keywords so it can make live predictions.
- You can refine mappings, retrain with MO-CCI data, and expand hazard buckets/controls as data grows.

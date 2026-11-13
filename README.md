# AI-Powered Marketing Intelligence Platform

End-to-end ML platform: KMeans customer segmentation + XGBoost campaign-acceptance prediction.

## Pipeline
data/raw -> src/preprocess.py (features) -> src/train.py (MLflow-tracked models)
       -> models/*.pkl -> backend (FastAPI) -> frontend (Streamlit)

## Local dev
1. `pip install -r requirements.txt`
2. `python -m src.train`              # trains & logs to MLflow
3. `mlflow ui --backend-store-uri file:./mlruns`   # http://127.0.0.1:5000
4. `uvicorn backend.main:app --reload --port 8000` # http://127.0.0.1:8000/docs
5. `streamlit run frontend/app.py`                 # http://127.0.0.1:8501

## Docker
`docker compose up --build`
Frontend: http://localhost:8501  |  API: http://localhost:8000/docs
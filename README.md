# Pima Diabetes Prediction Deployment

This repository contains a machine learning project for predicting diabetes using the **Pima Indians Diabetes Dataset**. The project includes:

- A **FastAPI model API** (`api`)
- A **Streamlit web application** (`app`) that interacts with the API
- Docker and Docker Compose setup for easy deployment

---

## Getting Started

### Requirements

- Python 3.11+
- Docker & Docker Compose (optional for containerized deployment)
- pip

---

### 1. Run Locally without Docker

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r code/deployment/api/requirements.txt
pip install -r code/deployment/app/requirements.txt
```

3. Start the FastAPI API:

```bash
uvicorn main:app --reload --app-dir code/deployment/api
```

- API runs at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

4. Start the Streamlit app:

```bash
streamlit run code/deployment/app/app.py
```

- App runs at [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

### 2. Run with Docker Compose

From `code/deployment` directory:

```bash
docker-compose up --build
```

- FastAPI API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Streamlit app: [http://127.0.0.1:8501](http://127.0.0.1:8501)

Stop containers:

```bash
docker-compose down
```

---

## How It Works

- **API**: Receives input features (age, BMI, glucose, etc.), loads the trained Random Forest model (`rf_model.pkl`) and returns predictions and probabilities.
- **Streamlit App**: Provides a simple UI for users to input features and get real-time predictions from the API.

---

## Notes

- Model file `rf_model.pkl` must be present in `models/` folder for both local run and Docker volume mounting.
- Docker Compose mounts `models/` into the API container automatically.
- API uses FastAPI and supports `/predict` endpoint.
- App uses Streamlit and communicates with the API via HTTP requests.

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

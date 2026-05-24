# Sentiments Analyser

Emotion analysis app split into a Vite + Tailwind frontend, a thin Express backend, and a Python ML service.

## Structure

- `frontend/` React UI built with Vite and Tailwind CSS
- `backend/` Express API that proxies requests to the Python service
- `ml-service/` Flask prediction service that loads the trained model
- `models_folder/` trained model artifacts used by the ML service
- `Datasets/` training and testing data used by the ML workflow

## Run locally

Start each service in its own terminal, in this order. The backend expects the ML service at `http://127.0.0.1:8000`.

### 1. ML service

```powershell
cd ml-service
.\\.venv\\Scripts\\python.exe app.py
```

Confirm it is running at `http://127.0.0.1:8000/health`.

### 2. Express backend

```powershell
cd backend
npm start
```

### 3. Frontend

```powershell
cd frontend
npm run dev
```

The frontend sends prediction requests to `/api/predict`, which is proxied by the backend and then forwarded to the ML service.

## Production environment variables

For a deployed frontend, set `VITE_API_URL` to the public URL of the Express backend. For the deployed backend, set `ML_SERVICE_URL` to the public URL of the Flask ML service. Copy the `.env.example` files when creating local `.env` files; never commit actual secrets.

Recommended production start command for the ML service:

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

The trained model artifacts were created with scikit-learn `1.6.1`. Keep the pinned dependency and Python runtime in `ml-service/` aligned with the model files to avoid pickle compatibility errors.

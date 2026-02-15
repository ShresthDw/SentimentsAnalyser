# Sentiments Analyser

Emotion analysis app split into a Vite + Tailwind frontend, a thin Express backend, and a Python ML service.

## Structure

- `frontend/` React UI built with Vite and Tailwind CSS
- `backend/` Express API that proxies requests to the Python service
- `ml-service/` Flask prediction service that loads the trained model
- `models_folder/` trained model artifacts used by the ML service
- `Datasets/` training and testing data used by the ML workflow

## Run locally

1. Start the ML service from `ml-service/`
2. Start the Express backend from `backend/`
3. Start the frontend from `frontend/`

The frontend sends prediction requests to `/api/predict`, which is proxied by the backend and then forwarded to the ML service.
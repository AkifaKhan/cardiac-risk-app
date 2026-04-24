# Cardiac Mortality Risk Predictor

In-hospital mortality prediction for cardiac surgery patients.

---

## Project Structure

```
cardiac-risk-app/
├── train.py          ← Step 1: Build & save the ML model
├── app.py            ← Step 2: FastAPI backend (the API)
├── index.html        ← Step 3: Frontend (the form clinicians use)
├── requirements.txt  ← Python dependencies
├── data.csv          ← YOUR DATA FILE (add this yourself)
├── model.pkl         ← Auto-generated after training
└── feature_info.json ← Auto-generated after training
```

---

## Setup (do this once)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Step 1 — Train the model

Put your data file in this folder as `data.csv` (tab-separated).
If your file uses commas, open train.py and change `sep="\t"` to `sep=","`.

```bash
python train.py
```

This prints model performance (AUC, confusion matrix, top features)
and saves `model.pkl` and `feature_info.json`.

---

## Step 2 — Start the API server

```bash
uvicorn app:app --reload
```

The API is now running at http://localhost:8000

---

## Step 3 — Open the app

Open your browser and go to:

```
http://localhost:8000
```

Fill in the patient form and press **Calculate Mortality Risk**.

---

## API Endpoints

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/`            | GET    | Frontend HTML                      |
| `/predict`     | POST   | Get risk prediction (JSON in/out)  |
| `/model-info`  | GET    | Model AUC, CV scores               |
| `/health`      | GET    | Health check                       |
| `/docs`        | GET    | Auto-generated API docs (Swagger)  |

---

## Deploying to Render (free hosting)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt && python train.py`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port 10000`
   - Environment: Python 3
5. Click Deploy → you get a public HTTPS URL

---

## Risk Stratification

| Risk Level | Probability   | Action                                          |
|------------|---------------|-------------------------------------------------|
| Low        | < 5%          | Standard post-op monitoring                    |
| Moderate   | 5% – 15%      | Enhanced monitoring, early intervention ready  |
| High       | 15% – 35%     | ICU-level care, cardiology review              |
| Critical   | > 35%         | Multidisciplinary review, reconsider surgery   |

---

⚠️ **Disclaimer**: This tool is for clinical decision *support* only.
It does not replace physician judgement.

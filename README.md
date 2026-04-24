# 🩺 GlycoSense — Diabetes Risk Prediction

A full-stack, production-ready web application for diabetes risk prediction using the **Pima Indians Diabetes Dataset** and three machine learning models.

---

## ✨ Features

- **3 ML Models**: Random Forest · Logistic Regression · Gradient Boosting
- **REST API**: Flask backend with CORS support
- **Single-file Frontend**: Dark-themed vanilla JS, no build step
- **Analytics Dashboard**: 5 dark-themed matplotlib/seaborn visualizations
- **Comparison Mode**: Side-by-side results from all 3 models
- **Pre-trained Models**: `.pkl` files included — no re-training required

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd glycosense
pip3 install -r backend/requirements.txt
```

### 2. Train models (generates `.pkl` files and plots)

```bash
python3 train_models.py
```

This creates:
- `backend/models/random_forest.pkl`
- `backend/models/logistic_regression.pkl`
- `backend/models/gradient_boosting.pkl`
- `backend/models/scaler.pkl`
- `backend/models/model_metrics.json`
- `backend/static/plots/*.png` (5 visualizations)

### 3. Start the API server

```bash
python3 backend/app.py
```

The API will be available at `http://localhost:8080`.

### 4. Open the frontend

Open `frontend/index.html` in any browser (double-click or `open frontend/index.html`).

---

## 🐚 One-liner (after training)

```bash
python3 backend/app.py &
open frontend/index.html
```

Or use the provided script:

```bash
chmod +x start.sh
./start.sh
```

---

## 📡 API Reference

### `GET /api/health`
Health check.

**Response:**
```json
{ "status": "ok", "models_loaded": 3, "message": "GlycoSense API is running" }
```

---

### `GET /api/models`
List available models with live AUC and accuracy.

**Response:**
```json
{
  "models": [
    { "key": "random_forest", "name": "Random Forest", "icon": "🌲",
      "auc": 0.845, "accuracy": 0.779, "loaded": true }
  ]
}
```

---

### `POST /api/predict`
Run prediction with a single model.

**Request:**
```json
{
  "model": "random_forest",
  "features": {
    "Pregnancies": 3,
    "Glucose": 130,
    "BloodPressure": 76,
    "SkinThickness": 28,
    "Insulin": 100,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 38
  }
}
```

**Response:**
```json
{
  "model": "random_forest",
  "prediction": 1,
  "label": "Diabetic",
  "probability": 0.7234,
  "risk_level": "High",
  "features": { ... }
}
```

- `model`: `random_forest` | `logistic_regression` | `gradient_boosting`
- `risk_level`: `Low` (< 35%) | `Moderate` (35–65%) | `High` (> 65%)

---

### `POST /api/compare`
Compare all 3 models simultaneously.

**Request:** Same `features` object as `/api/predict` (no `model` field needed).

**Response:**
```json
{
  "results": {
    "random_forest":       { "prediction": 1, "label": "Diabetic", "probability": 0.72, "risk_level": "High" },
    "logistic_regression": { "prediction": 1, "label": "Diabetic", "probability": 0.68, "risk_level": "High" },
    "gradient_boosting":   { "prediction": 0, "label": "Non-Diabetic", "probability": 0.49, "risk_level": "Moderate" }
  }
}
```

---

### `GET /api/metrics`
Retrieve evaluation metrics for all models.

**Response:**
```json
{
  "metrics": {
    "random_forest": {
      "auc": 0.845, "accuracy": 0.779, "precision": 0.712,
      "recall": 0.654, "f1": 0.682
    }
  }
}
```

---

### `GET /api/plots`
List available plot filenames and URLs.

### `GET /api/plots/<filename>`
Serve a specific plot PNG file.

---

## 📊 Dataset

**Pima Indians Diabetes Database**

| Feature | Description | Unit |
|---------|-------------|------|
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose concentration (2h OGTT) | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skin fold thickness | mm |
| Insulin | 2-hour serum insulin | mu U/ml |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Diabetes heredity score | score |
| Age | Age of patient | years |
| Outcome | 0 = Non-Diabetic, 1 = Diabetic | — |

**Preprocessing**: Biologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI are replaced with column medians.

---

## 📁 Project Structure

```
glycosense/
├── train_models.py           ← Training script (run first)
├── start.sh                  ← One-click launcher
├── README.md
│
├── backend/
│   ├── app.py                ← Flask API (port 8080)
│   ├── requirements.txt
│   ├── models/               ← Auto-created by train_models.py
│   │   ├── random_forest.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── scaler.pkl
│   │   └── model_metrics.json
│   └── static/plots/         ← Auto-created visualizations
│       ├── feature_distributions.png
│       ├── feature_boxplots.png
│       ├── correlation_heatmap.png
│       ├── roc_curves.png
│       └── feature_importance.png
│
└── frontend/
    └── index.html            ← Open in browser
```

---

## ⚙️ Technical Notes

- Backend runs on **port 8080** (avoids macOS AirPlay conflict on 5000)
- All file paths use `os.path.dirname(os.path.abspath(__file__))` — no hardcoded paths
- Random Forest predicts on raw features; LR and GBM use `StandardScaler`
- `pandas.DataFrame` used for predictions (suppresses sklearn feature-name warnings)
- CORS headers added to all responses for `file://` frontend access

---

## 📜 License

For educational and research use. Dataset originally from the UCI ML Repository.

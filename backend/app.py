"""
GlycoSense — Diabetes Risk Prediction API
Flask backend serving predictions and analytics
"""

import os
import json
import glob

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, send_file

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
PLOTS_DIR     = os.path.join(BASE_DIR, 'static', 'plots')
FRONTEND_DIR  = os.path.join(BASE_DIR, '..', 'frontend')

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# ─── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')

# ─── Load Models ───────────────────────────────────────────────────────────────
print("🔧 Loading models...")
try:
    scaler              = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    random_forest       = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    logistic_regression = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    gradient_boosting   = joblib.load(os.path.join(MODELS_DIR, 'gradient_boosting.pkl'))
    
    with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
        MODEL_METRICS = json.load(f)
    
    MODELS = {
        'random_forest':       random_forest,
        'logistic_regression': logistic_regression,
        'gradient_boosting':   gradient_boosting,
    }
    print("   ✅ All models loaded successfully")
except FileNotFoundError as e:
    print(f"   ❌ Model files not found: {e}")
    print("   ▶  Run `python3 train_models.py` first to train the models.")
    MODELS = {}
    MODEL_METRICS = {}
    scaler = None

# ─── CORS Helper ───────────────────────────────────────────────────────────────
def cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.after_request
def add_cors(response):
    return cors(response)

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()
        return cors(resp)

# ─── Frontend ──────────────────────────────────────────────────────────────────

@app.route('/')
def serve_frontend():
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))

# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({
        'status':  'ok',
        'models_loaded': len(MODELS),
        'message': 'GlycoSense API is running'
    })


@app.route('/api/models')
def list_models():
    available = []
    display = {
        'random_forest':       {'name': 'Random Forest',       'icon': '🌲', 'description': 'Ensemble of decision trees — robust to outliers and non-linear patterns.'},
        'logistic_regression': {'name': 'Logistic Regression', 'icon': '📈', 'description': 'Linear probabilistic model — fast, interpretable baseline.'},
        'gradient_boosting':   {'name': 'Gradient Boosting',   'icon': '🚀', 'description': 'Sequential boosting ensemble — high accuracy via adaptive learning.'},
    }
    for key, info in display.items():
        entry = {**info, 'key': key, 'loaded': key in MODELS}
        if key in MODEL_METRICS:
            entry['auc']      = MODEL_METRICS[key]['auc']
            entry['accuracy'] = MODEL_METRICS[key]['accuracy']
        available.append(entry)
    return jsonify({'models': available})


@app.route('/api/metrics')
def get_metrics():
    return jsonify({'metrics': MODEL_METRICS})


@app.route('/api/plots')
def list_plots():
    files  = glob.glob(os.path.join(PLOTS_DIR, '*.png'))
    plots  = []
    labels = {
        'feature_distributions': 'Feature Distributions by Class',
        'feature_boxplots':      'Feature Boxplots',
        'correlation_heatmap':   'Correlation Heatmap',
        'roc_curves':            'ROC Curves',
        'feature_importance':    'Feature Importance',
    }
    for fp in sorted(files):
        fname = os.path.basename(fp)
        stem  = os.path.splitext(fname)[0]
        plots.append({
            'filename': fname,
            'label':    labels.get(stem, stem),
            'url':      f'/api/plots/{fname}',
        })
    return jsonify({'plots': plots})


@app.route('/api/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODELS:
        return jsonify({'error': 'Models not loaded. Run train_models.py first.'}), 503

    body = request.get_json(force=True, silent=True) or {}
    model_key = body.get('model', 'random_forest')
    features  = body.get('features', {})

    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400

    # Validate features
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    try:
        values = [float(features[f]) for f in FEATURE_NAMES]
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid feature value: {e}'}), 400

    # Build DataFrame for prediction (avoids sklearn feature-name warnings)
    row_df = pd.DataFrame([values], columns=FEATURE_NAMES)

    model = MODELS[model_key]

    if model_key == 'random_forest':
        prediction  = model.predict(row_df)[0]
        probability = float(model.predict_proba(row_df)[0][1])
    else:
        row_scaled = pd.DataFrame(scaler.transform(row_df), columns=FEATURE_NAMES)
        prediction  = model.predict(row_scaled)[0]
        probability = float(model.predict_proba(row_scaled)[0][1])

    if probability < 0.35:
        risk_level = 'Low'
    elif probability < 0.65:
        risk_level = 'Moderate'
    else:
        risk_level = 'High'

    return jsonify({
        'model':       model_key,
        'prediction':  int(prediction),
        'label':       'Diabetic' if prediction == 1 else 'Non-Diabetic',
        'probability': round(probability, 4),
        'risk_level':  risk_level,
        'features':    {k: v for k, v in zip(FEATURE_NAMES, values)},
    })


@app.route('/api/compare', methods=['POST'])
def compare():
    """Run prediction with all three models and return side-by-side results."""
    if not MODELS:
        return jsonify({'error': 'Models not loaded.'}), 503

    body     = request.get_json(force=True, silent=True) or {}
    features = body.get('features', {})

    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    try:
        values = [float(features[f]) for f in FEATURE_NAMES]
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid feature value: {e}'}), 400

    row_df     = pd.DataFrame([values], columns=FEATURE_NAMES)
    row_scaled = pd.DataFrame(scaler.transform(row_df), columns=FEATURE_NAMES)

    results = {}
    for key, model in MODELS.items():
        if key == 'random_forest':
            prob = float(model.predict_proba(row_df)[0][1])
        else:
            prob = float(model.predict_proba(row_scaled)[0][1])

        pred = 1 if prob >= 0.5 else 0
        risk = 'Low' if prob < 0.35 else ('Moderate' if prob < 0.65 else 'High')
        results[key] = {
            'prediction':  pred,
            'label':       'Diabetic' if pred == 1 else 'Non-Diabetic',
            'probability': round(prob, 4),
            'risk_level':  risk,
        }

    return jsonify({'results': results, 'features': dict(zip(FEATURE_NAMES, values))})


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🩺 GlycoSense API starting on http://localhost:8080\n")
    app.run(host='0.0.0.0', port=8080, debug=False)

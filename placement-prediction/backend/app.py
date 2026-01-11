from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)

# ===============================
# GLOBALS (LAZY LOADED)
# ===============================
model = None
scaler = None
features = None
sectors = None


def load_models():
    """Load ML artifacts only once, when first needed"""
    global model, scaler, features, sectors

    if model is not None:
        return

    try:
        base_path = os.path.dirname(__file__)
        models_path = os.path.join(base_path, "models")

        model = joblib.load(os.path.join(models_path, "sector_classifier_rf_optimized.pkl"))
        scaler = joblib.load(os.path.join(models_path, "sector_scaler.pkl"))
        features = joblib.load(os.path.join(models_path, "sector_features_base.pkl"))
        sectors = joblib.load(os.path.join(models_path, "sector_labels.pkl"))

        print("✅ ML models loaded successfully")

    except Exception as e:
        print("❌ MODEL LOADING FAILED")
        traceback.print_exc()
        raise RuntimeError("Model loading failed") from e


# ===============================
# ROUTES
# ===============================

@app.route("/api/sector-predict", methods=["POST"])
def sector_predict():
    try:
        load_models()

        data = request.json or {}

        # Base features
        score800 = float(data.get('score800', 0))
        aptitude = float(data.get('aptitude', 0))
        english = float(data.get('english', 0))
        quantitative = float(data.get('quantitative', 0))
        analytical = float(data.get('analytical', 0))
        domain = float(data.get('domain', 0))
        comp_fund = float(data.get('comp_fund', 0))
        coding = float(data.get('coding', 0))
        personality = float(data.get('personality', 0))

        # Interaction features
        aptitude_quantitative = aptitude * quantitative
        english_analytical = english * analytical
        domain_coding = domain * coding

        input_data = [
            score800, aptitude, english, quantitative, analytical,
            domain, comp_fund, coding, personality,
            aptitude_quantitative, english_analytical, domain_coding
        ]

        X = np.array([input_data], dtype=np.float32)
        X_scaled = scaler.transform(X)

        probabilities = model.predict_proba(X_scaled)

        results = []

        if isinstance(probabilities, list):
            # MultiOutputClassifier
            for i, sector in enumerate(sectors):
                prob = float(probabilities[i][0, 1])
                results.append({
                    "sector": sector,
                    "probability": f"{prob*100:.1f}%",
                    "raw_score": prob
                })
        else:
            probs = probabilities[0]
            for i, sector in enumerate(sectors):
                prob = float(probs[i])
                results.append({
                    "sector": sector,
                    "probability": f"{prob*100:.1f}%",
                    "raw_score": prob
                })

        results.sort(key=lambda x: x["raw_score"], reverse=True)

        return jsonify({
            "success": True,
            "best_sector": results[0]["sector"],
            "best_probability": results[0]["probability"],
            "top_3": [
                {
                    "rank": i + 1,
                    "sector": r["sector"],
                    "probability": r["probability"]
                }
                for i, r in enumerate(results[:3])
            ],
            "all_sectors": [
                {"sector": r["sector"], "probability": r["probability"]}
                for r in results
            ]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# NOTE:
# ❌ DO NOT call app.run()
# Gunicorn controls the server on Render

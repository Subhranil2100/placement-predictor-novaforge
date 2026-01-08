from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys
import traceback

app = Flask(__name__)
CORS(app)

print("\n" + "="*70)
print("🔍 PLACEMENT PREDICTOR API - INITIALIZATION")
print("="*70)

# Check current directory
print(f"📍 Current directory: {os.getcwd()}")
print(f"📂 Contents: {os.listdir('.')}")

# Check if models folder exists
if not os.path.exists("models"):
    print("❌ ERROR: 'models' folder NOT found!")
    print("   Please create 'models' folder with pickle files")
    sys.exit(1)

print("✅ 'models' folder found")
model_files = os.listdir('models')
print(f"📦 Files in models/: {model_files}\n")

# Load models with detailed error reporting
models_loaded = True

try:
    print("   Loading: sector_classifier_rf_optimized.pkl...")
    model = joblib.load("models/sector_classifier_rf_optimized.pkl")
    print("   ✅ Model loaded successfully")
    print(f"      Model type: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    models_loaded = False

try:
    print("   Loading: sector_scaler.pkl...")
    scaler = joblib.load("models/sector_scaler.pkl")
    print("   ✅ Scaler loaded successfully")
    print(f"      Expected features: {scaler.n_features_in_}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    models_loaded = False

try:
    print("   Loading: sector_features_base.pkl...")
    features = joblib.load("models/sector_features_base.pkl")
    print(f"   ✅ Features loaded: {features}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    models_loaded = False

try:
    print("   Loading: sector_labels.pkl...")
    sectors = joblib.load("models/sector_labels.pkl")
    print(f"   ✅ Sectors loaded: {sectors}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    models_loaded = False

if not models_loaded:
    print("\n❌ FATAL: Some model files are missing!")
    print("   Solution: Ensure all .pkl files are in models/ folder")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print(f"📊 Ready to serve predictions for {len(sectors)} sectors")
print(f"🤖 Model Type: {type(model).__name__}")
print("="*70 + "\n")

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/sector-predict', methods=['POST', 'OPTIONS'])
def sector_predict():
    """Main prediction endpoint with 12 engineered features"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        print(f"\n📥 Prediction request received")
        print(f"   Raw data: {data}")
        
        # Extract 9 BASE input features
        score800 = float(data.get('score800', 0))
        aptitude = float(data.get('aptitude', 0))
        english = float(data.get('english', 0))
        quantitative = float(data.get('quantitative', 0))
        analytical = float(data.get('analytical', 0))
        domain = float(data.get('domain', 0))
        comp_fund = float(data.get('comp_fund', 0))
        coding = float(data.get('coding', 0))
        personality = float(data.get('personality', 0))
        
        # CREATE 3 INTERACTION FEATURES (same as training)
        aptitude_quantitative = aptitude * quantitative
        english_analytical = english * analytical
        domain_coding = domain * coding
        
        # COMBINE ALL 12 FEATURES (9 base + 3 interactions)
        input_data = [
            score800,
            aptitude,
            english,
            quantitative,
            analytical,
            domain,
            comp_fund,
            coding,
            personality,
            aptitude_quantitative,
            english_analytical,
            domain_coding
        ]
        
        print(f"   Base features (9): {input_data[:9]}")
        print(f"   Interactions (3): {input_data[9:]}")
        print(f"   Total features: {len(input_data)}")
        print(f"   Scaler expects: {scaler.n_features_in_} features")
        
        # Prepare and scale
        X = np.array([input_data], dtype=np.float32)
        print(f"   Input array shape: {X.shape}")
        
        X_scaled = scaler.transform(X)
        print(f"   ✅ Features scaled successfully. Shape: {X_scaled.shape}")
        
        # Get probability estimates
        print(f"\n🎯 GETTING PREDICTIONS:")
        probabilities = model.predict_proba(X_scaled)
        print(f"   Proba type: {type(probabilities).__name__}")
        print(f"   Proba is list: {isinstance(probabilities, list)}")
        
        if isinstance(probabilities, list):
            print(f"   List length: {len(probabilities)}")
            if len(probabilities) > 0:
                print(f"   First element shape: {probabilities[0].shape}")
        else:
            print(f"   Array shape: {probabilities.shape}")
        
        # ✅ KEY FIX: Handle MultiOutputClassifier
        # MultiOutputClassifier returns a LIST of arrays
        # Each array has shape (1, 2) with [negative_prob, positive_prob]
        # We need the positive class probability (index 1)
        
        results = []
        
        if isinstance(probabilities, list):
            # This is MultiOutputClassifier - list of arrays
            print(f"   ✅ Detected MultiOutputClassifier format (list of arrays)")
            
            for i, sector in enumerate(sectors):
                if i < len(probabilities):
                    sector_proba = probabilities[i]  # shape: (1, 2)
                    # Get positive class probability [0, 1]
                    prob = float(sector_proba[0, 1])
                else:
                    prob = 0.0
                
                results.append({
                    'sector': sector,
                    'probability': f"{prob*100:.1f}%",
                    'raw_score': prob
                })
        else:
            # Regular classifier - 2D array
            print(f"   ✅ Detected regular classifier format (2D array)")
            
            if len(probabilities.shape) == 1:
                probs = probabilities
            else:
                probs = probabilities[0]
            
            for i, sector in enumerate(sectors):
                if i < len(probs):
                    prob = float(probs[i])
                else:
                    prob = 0.0
                
                results.append({
                    'sector': sector,
                    'probability': f"{prob*100:.1f}%",
                    'raw_score': prob
                })
        
        # Sort by score
        results.sort(key=lambda x: x['raw_score'], reverse=True)
        
        print(f"\n✅ Top 3 predictions:")
        for i, r in enumerate(results[:3]):
            print(f"   #{i+1}: {r['sector']} - {r['probability']}")
        
        response = {
            'success': True,
            'best_sector': results[0]['sector'],
            'best_probability': results[0]['probability'],
            'top_3': [
                {
                    'rank': i+1,
                    'sector': r['sector'],
                    'probability': r['probability']
                }
                for i, r in enumerate(results[:3])
            ],
            'all_sectors': [
                {
                    'sector': r['sector'],
                    'probability': r['probability']
                }
                for r in results
            ],
            'message': f"Best fit: {results[0]['sector']} with {results[0]['probability']} confidence"
        }
        
        print(f"\n✅ Response sent successfully\n")
        return jsonify(response)
        
    except Exception as e:
        print(f"\n❌ ERROR in sector_predict:")
        print(f"   {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'sectors': len(sectors),
        'base_features': len(features),
        'total_features_with_interactions': 12,
        'model_type': type(model).__name__,
        'model_loaded': True,
        'message': 'API is running and ready for predictions'
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("🚀 Starting Flask API server...")
    print("📡 Listening on http://127.0.0.1:5001")
    print("🔗 Endpoint: POST /api/sector-predict")
    print("💚 Health check: GET /api/health")
    print("\n✅ System ready! 9 base features → 12 engineered features → 10 sector predictions\n")
    print("Press CTRL+C to stop the server\n")
    app.run(debug=True, port=5001, host='127.0.0.1')

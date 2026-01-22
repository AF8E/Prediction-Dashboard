"""
Phase 3: Flask Dashboard Application
Main application file for infrastructure failure prediction dashboard
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
# SECURITY: Change this to a secure random key in production
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Global variables for models
regression_model = None
classification_model = None
scaler = None
label_encoders = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load trained models and preprocessing objects"""
    global regression_model, classification_model, scaler, label_encoders
    
    models_dir = Path('models')
    
    try:
        regression_model = joblib.load(models_dir / 'failure_model.pkl')
        classification_model = joblib.load(models_dir / 'classifier_model.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        label_encoders = joblib.load(models_dir / 'label_encoders.pkl')
        
        print("✓ Models loaded successfully")
        return True
    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        print("Please run Phase 2 (src/train_model.py) to train models first.")
        return False
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


def get_summary_stats():
    """Get summary statistics from master data"""
    try:
        master_data = pd.read_csv('data/Master_Training_Data.csv')
        
        stats = {
            'total_facilities': len(master_data),
            'high_risk_count': len(master_data[master_data['risk_score'] > 70]),
            'avg_condition_index': master_data['Condition Index'].mean() if 'Condition Index' in master_data else 65,
            'total_failures': master_data['total_failures'].sum() if 'total_failures' in master_data else 0,
            'avg_age': master_data['Age (years)'].mean() if 'Age (years)' in master_data else 0
        }
        
        return stats
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            'total_facilities': 0,
            'high_risk_count': 0,
            'avg_condition_index': 0,
            'total_failures': 0,
            'avg_age': 0
        }


def get_degradation_projection():
    """Get projected degradation data for chart"""
    try:
        master_data = pd.read_csv('data/Master_Training_Data.csv')
        
        # Calculate average condition by age group
        if 'Age (years)' in master_data.columns and 'Condition Index' in master_data.columns:
            master_data['age_group'] = pd.cut(
                master_data['Age (years)'],
                bins=[0, 10, 20, 30, 40, 50, 60],
                labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
            )
            
            projection = master_data.groupby('age_group')['Condition Index'].mean()
            
            return {
                'labels': [str(x) for x in projection.index],
                'values': [round(x, 1) for x in projection.values]
            }
        else:
            # Default projection
            return {
                'labels': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'],
                'values': [95, 85, 75, 65, 55, 45]
            }
    except Exception as e:
        print(f"Error getting projection: {e}")
        return {
            'labels': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'],
            'values': [95, 85, 75, 65, 55, 45]
        }


def prepare_features_for_prediction(df):
    """
    Prepare features from uploaded data for prediction
    
    Args:
        df: Uploaded facility dataframe
        
    Returns:
        Feature matrix ready for prediction
    """
    # Define expected features (must match training)
    numeric_features = [
        'Age (years)',
        'Life Expectancy',
        'total_failures',
        'failure_rate',
        'condition_delta',
        'risk_score',
        'days_between_failures'
    ]
    
    categorical_features = ['most_common_failure']
    
    # Add optional features if they exist
    optional_features = [
        'avg_system_condition',
        'min_system_condition',
        'system_condition_std',
        'avg_system_life_expectancy'
    ]
    
    # Calculate derived features if not present
    if 'total_failures' not in df.columns:
        df['total_failures'] = 0
    
    if 'failure_rate' not in df.columns and 'Age (years)' in df.columns:
        df['failure_rate'] = df['total_failures'] / df['Age (years)'].clip(lower=1)
    
    if 'condition_delta' not in df.columns and 'Condition Index' in df.columns:
        df['condition_delta'] = 100 - df['Condition Index']
    
    if 'risk_score' not in df.columns:
        df['risk_score'] = (
            df.get('failure_rate', 0) * 0.3 +
            df.get('condition_delta', 35) * 0.4 +
            (df.get('Age (years)', 0) / df.get('Life Expectancy', 50)) * 100 * 0.3
        )
    
    if 'days_between_failures' not in df.columns:
        df['days_between_failures'] = 365
    
    if 'most_common_failure' not in df.columns:
        df['most_common_failure'] = 'none'
    
    # Build feature list
    all_features = []
    for feat in numeric_features:
        if feat in df.columns:
            all_features.append(feat)
            df[feat] = df[feat].fillna(df[feat].median() if df[feat].notna().any() else 0)
    
    for feat in optional_features:
        if feat in df.columns:
            all_features.append(feat)
            df[feat] = df[feat].fillna(df[feat].median() if df[feat].notna().any() else 0)
    
    # Encode categorical features
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df[f'{col}_encoded'] = 0
            
            all_features.append(f'{col}_encoded')
    
    # Select features
    X = df[all_features].copy()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, all_features


def predict_failure_dates(df):
    """
    Predict failure dates and costs for facilities
    
    Args:
        df: Facility dataframe
        
    Returns:
        DataFrame with predictions
    """
    # Prepare features
    X, features = prepare_features_for_prediction(df)
    
    # Predict condition index
    predicted_condition = regression_model.predict(X)
    
    # Predict failure probability
    failure_probability = classification_model.predict_proba(X)[:, 1]
    
    # Calculate predicted failure date
    current_age = df.get('Age (years)', 20)
    life_expectancy = df.get('Life Expectancy', 50)
    current_condition = df.get('Condition Index', predicted_condition)
    
    # Estimate years until failure based on degradation rate
    degradation_rate = (100 - predicted_condition) / current_age.clip(lower=1)
    years_until_critical = (predicted_condition - 25) / degradation_rate.clip(lower=0.1)
    
    # Adjust based on failure probability
    years_until_failure = years_until_critical * (1 - failure_probability * 0.5)
    
    predicted_failure_date = pd.to_datetime('today') + pd.to_timedelta(years_until_failure * 365, unit='D')
    
    # Estimate cost (simplified model)
    base_replacement_cost = {
        'Foundation': 500000,
        'Basement': 300000,
        'Superstructure': 400000,
        'Roofing': 150000,
        'HVAC': 200000,
        'Electric': 180000,
        'Plumbing': 120000
    }
    
    facility_type = df.get('Type', 'Unknown')
    estimated_cost = facility_type.apply(
        lambda x: base_replacement_cost.get(x, 250000) * (1 + (100 - predicted_condition) / 100)
    )
    
    # Create results dataframe
    results = df.copy()
    results['Predicted_Condition_Index'] = predicted_condition
    results['Failure_Probability'] = (failure_probability * 100).round(1)
    results['Predicted_Failure_Date'] = predicted_failure_date
    results['Years_Until_Failure'] = years_until_failure.round(1)
    results['Estimated_Replacement_Cost'] = estimated_cost.round(0)
    results['Risk_Level'] = pd.cut(
        failure_probability * 100,
        bins=[0, 30, 60, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    return results


@app.route('/')
def index():
    """Dashboard home page"""
    stats = get_summary_stats()
    projection = get_degradation_projection()
    
    return render_template(
        'index.html',
        stats=stats,
        projection_labels=json.dumps(projection['labels']),
        projection_values=json.dumps(projection['values'])
    )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """File upload page and handler"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Security: Check file extension
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload .csv, .xlsx, or .xls files only.', 'error')
            return redirect(request.url)
        
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                flash(f'File "{filename}" uploaded successfully!', 'success')
                return redirect(url_for('predict', filename=filename))
            except Exception as e:
                flash(f'Error saving file: {str(e)}', 'error')
                return redirect(request.url)
    
    return render_template('upload.html')


@app.route('/predict')
def predict():
    """Prediction results page"""
    filename = request.args.get('filename')
    
    if not filename:
        flash('No file specified', 'error')
        return redirect(url_for('upload'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('File not found', 'error')
        return redirect(url_for('upload'))
    
    try:
        # Load uploaded file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            flash('Unsupported file format', 'error')
            return redirect(url_for('upload'))
        
        # Make predictions
        results = predict_failure_dates(df)
        
        # Convert to dict for template
        results_dict = results.to_dict('records')
        
        # Summary statistics
        summary = {
            'total_facilities': len(results),
            'high_risk': len(results[results['Risk_Level'] == 'High']),
            'medium_risk': len(results[results['Risk_Level'] == 'Medium']),
            'low_risk': len(results[results['Risk_Level'] == 'Low']),
            'avg_failure_probability': results['Failure_Probability'].mean(),
            'total_estimated_cost': results['Estimated_Replacement_Cost'].sum()
        }
        
        return render_template(
            'results.html',
            results=results_dict,
            summary=summary,
            filename=filename
        )
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        results = predict_failure_dates(df)
        
        # Return as JSON
        return jsonify({
            'success': True,
            'predictions': results.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    models_loaded = all([
        regression_model is not None,
        classification_model is not None,
        scaler is not None
    ])
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'degraded',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("="*60)
    print("INFRASTRUCTURE FAILURE PREDICTION DASHBOARD")
    print("="*60)
    
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠ WARNING: Models not loaded!")
        print("Please run Phase 2 (train_model.py) to train models first.")
        print("The application will start but predictions will not work.\n")
    
    print("\nStarting Flask server...")
    print("Dashboard URL: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
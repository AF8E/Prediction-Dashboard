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


def predict_failure_dates(filepath_or_df):
    """
    Predict failure dates and costs for systems, then aggregate to facilities
    
    Args:
        filepath_or_df: File path (str) or DataFrame
        
    Returns:
        Dictionary with hierarchical structure: {'facilities': [...]}
    """
    # Load data - handle both file path and DataFrame
    if isinstance(filepath_or_df, str):
        if filepath_or_df.endswith('.csv'):
            df = pd.read_csv(filepath_or_df)
            # For CSV, assume it's already the system data
            systems_df = df
            facility_df = pd.DataFrame()  # Empty, will infer from systems
        else:
            # Excel file - load both sheets
            try:
                facility_df = pd.read_excel(filepath_or_df, sheet_name='Facility Condition Index')
            except:
                facility_df = pd.DataFrame()
            
            try:
                systems_df = pd.read_excel(filepath_or_df, sheet_name='System Condition Index')
            except:
                # Fallback: use first sheet as systems
                systems_df = pd.read_excel(filepath_or_df, sheet_name=0)
    else:
        # DataFrame provided
        systems_df = filepath_or_df
        facility_df = pd.DataFrame()
    
    # Group systems by facility
    # Identify facility ID column in systems
    facility_id_col = None
    for col in ['Facility Number', 'Facility ID', 'Parent Facility', 'Facility', 'facility_id']:
        if col in systems_df.columns:
            facility_id_col = col
            break
    
    # If no facility column, try to infer from system ID
    if facility_id_col is None:
        system_id_col = None
        for col in ['Unique Identifier', 'System ID', 'ID', 'entity_id']:
            if col in systems_df.columns:
                system_id_col = col
                break
        
        if system_id_col:
            # Heuristic: extract facility from system ID
            def extract_facility(system_id):
                if pd.isna(system_id):
                    return None
                import re
                match = re.search(r'Facility\s*#?(\d+)', str(system_id), re.IGNORECASE)
                if match:
                    return f"FAC-{int(match.group(1)):03d}"
                match = re.search(r'(\d+)', str(system_id))
                if match:
                    sys_num = int(match.group(1))
                    facility_num = ((sys_num - 1) // 15) + 1
                    return f"FAC-{facility_num:03d}"
                return None
            
            systems_df['facility_id'] = systems_df[system_id_col].apply(extract_facility)
            facility_id_col = 'facility_id'
        else:
            # Default grouping
            systems_per_facility = 15
            systems_df['facility_id'] = (systems_df.index // systems_per_facility).apply(
                lambda x: f"FAC-{x+1:03d}"
            )
            facility_id_col = 'facility_id'
    
    # Get system ID column
    system_id_col = None
    for col in ['Unique Identifier', 'System ID', 'ID', 'entity_id']:
        if col in systems_df.columns:
            system_id_col = col
            break
    
    if system_id_col is None:
        systems_df['entity_id'] = systems_df.index.map(lambda x: f"SYS-{x+1:03d}")
        system_id_col = 'entity_id'
    
    # Prepare features for each system and predict
    system_predictions = []
    
    for idx, system in systems_df.iterrows():
        # Create single-row DataFrame for prediction
        system_df = pd.DataFrame([system])
        
        # Prepare features
        X, features = prepare_features_for_prediction(system_df)
        
        # Predict condition index
        predicted_condition = regression_model.predict(X)[0]
        
        # Predict failure probability
        failure_probability = classification_model.predict_proba(X)[0, 1]
        
        # Calculate predicted failure date
        current_age = system.get('Age (years)', 20)
        life_expectancy = system.get('Life Expectancy', 50)
        current_condition = system.get('Condition Index', predicted_condition)
        
        # Estimate years until failure
        degradation_rate = (100 - predicted_condition) / max(current_age, 1)
        years_until_critical = (predicted_condition - 25) / max(degradation_rate, 0.1)
        years_until_failure = years_until_critical * (1 - failure_probability * 0.5)
        
        predicted_failure_date = pd.to_datetime('today') + pd.to_timedelta(years_until_failure * 365, unit='D')
        
        # Estimate cost
        base_replacement_cost = {
            'Foundation': 500000,
            'Basement': 300000,
            'Superstructure': 400000,
            'Roofing': 150000,
            'HVAC': 200000,
            'Electric': 180000,
            'Plumbing': 120000,
            'Fire Protection': 100000
        }
        
        system_type = system.get('Type', system.get('System Type', 'Unknown'))
        estimated_cost = base_replacement_cost.get(system_type, 250000) * (1 + (100 - predicted_condition) / 100)
        
        # Determine risk level
        risk_prob = failure_probability * 100
        if risk_prob >= 60:
            risk_level = 'High'
        elif risk_prob >= 30:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        system_predictions.append({
            'system_id': system.get(system_id_col, f"SYS-{idx+1:03d}"),
            'system_type': system_type,
            'facility_id': system.get(facility_id_col, f"FAC-{(idx // 15) + 1:03d}"),
            'system_condition': float(current_condition),
            'predicted_condition': float(predicted_condition),
            'failure_probability': float(risk_prob),
            'failure_date': predicted_failure_date.strftime('%Y-%m-%d'),
            'cost': float(estimated_cost),
            'risk': risk_level
        })
    
    # Group systems by facility and aggregate
    facilities_dict = {}
    
    for sys_pred in system_predictions:
        fac_id = sys_pred['facility_id']
        
        if fac_id not in facilities_dict:
            # Get facility info if available
            fac_info = {}
            if len(facility_df) > 0 and 'facility_id' in facility_df.columns:
                fac_row = facility_df[facility_df['facility_id'] == fac_id]
                if len(fac_row) > 0:
                    fac_info = fac_row.iloc[0].to_dict()
            
            facilities_dict[fac_id] = {
                'facility_id': fac_id,
                'facility_name': fac_info.get('Title', fac_info.get('Name', fac_id)),
                'systems': []
            }
        
        facilities_dict[fac_id]['systems'].append(sys_pred)
    
    # Aggregate facility-level metrics
    facilities_list = []
    
    for fac_id, facility_data in facilities_dict.items():
        systems = facility_data['systems']
        
        # Weighted average condition (by life expectancy - use system types as proxy)
        system_conditions = [s['system_condition'] for s in systems]
        facility_condition = sum(system_conditions) / len(system_conditions) if system_conditions else 65
        
        # Max failure probability (facility fails when worst system fails)
        facility_failure_probability = max([s['failure_probability'] for s in systems]) if systems else 0
        
        # Min failure date (facility fails when first system fails)
        failure_dates = [pd.to_datetime(s['failure_date']) for s in systems]
        facility_failure_date = min(failure_dates).strftime('%Y-%m-%d') if failure_dates else None
        
        # Sum of costs
        facility_cost = sum([s['cost'] for s in systems])
        
        # Risk level: High if ANY system is High, Medium if ANY is Medium
        system_risks = [s['risk'] for s in systems]
        if 'High' in system_risks:
            facility_risk = 'High'
        elif 'Medium' in system_risks:
            facility_risk = 'Medium'
        else:
            facility_risk = 'Low'
        
        facilities_list.append({
            'facility_id': fac_id,
            'facility_name': facility_data['facility_name'],
            'facility_condition': round(facility_condition, 1),
            'facility_failure_probability': round(facility_failure_probability, 1),
            'facility_failure_date': facility_failure_date,
            'facility_cost': round(facility_cost, 0),
            'facility_risk': facility_risk,
            'systems': systems
        })
    
    return {'facilities': facilities_list}


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
        # Make predictions (function handles file loading)
        hierarchical_results = predict_failure_dates(filepath)
        facilities = hierarchical_results['facilities']
        
        # Calculate summary statistics from facilities
        summary = {
            'total_facilities': len(facilities),
            'high_risk': sum(1 for f in facilities if f['facility_risk'] == 'High'),
            'medium_risk': sum(1 for f in facilities if f['facility_risk'] == 'Medium'),
            'low_risk': sum(1 for f in facilities if f['facility_risk'] == 'Low'),
            'avg_failure_probability': sum(f['facility_failure_probability'] for f in facilities) / len(facilities) if facilities else 0,
            'total_estimated_cost': sum(f['facility_cost'] for f in facilities)
        }
        
        return render_template(
            'results.html',
            results=hierarchical_results,  # Pass full hierarchical structure
            summary=summary,
            filename=filename
        )
        
    except Exception as e:
        import traceback
        flash(f'Error processing file: {str(e)}', 'error')
        print(f"Error details: {traceback.format_exc()}")
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
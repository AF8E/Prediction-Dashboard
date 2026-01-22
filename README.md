# Infrastructure Failure Prediction Dashboard
## US Space Command Facility Management System

A Flask-based predictive analytics dashboard for forecasting infrastructure failures and managing facility maintenance for US Space Command installations. Uses machine learning (Random Forest) to predict condition degradation, failure probabilities, and replacement costs.

---

## 📋 Project Structure

```
Prediction-Dashboard/
├── app.py                          # Main Flask application (Phase 3)
├── requirements.txt                # Python dependencies
├── setup_and_run.bat              # Automated setup script (Windows)
├── README.md                       # This file
├── DEPLOYMENT.md                   # Deployment checklist
│
├── data/                           # Data directory
│   ├── generated_facility_data.xlsx    # Inventory data (input)
│   ├── Simulated_Data.xlsx            # Work order history (input)
│   └── Master_Training_Data.csv       # Processed training data (output)
│
├── src/                            # Source code
│   ├── data_prep.py               # Phase 1: ETL script with regex extraction
│   └── train_model.py             # Phase 2: ML model training
│
├── models/                         # Trained models (auto-generated)
│   ├── failure_model.pkl          # Regression model
│   ├── classifier_model.pkl       # Classification model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoders.pkl         # Categorical encoders
│   └── model_metadata.json        # Model metadata
│
├── templates/                      # HTML templates
│   ├── index.html                 # Dashboard home (Bootstrap 5 + Chart.js)
│   ├── upload.html                # File upload page
│   └── results.html               # Prediction results
│
├── static/                         # Static assets (optional)
│   ├── css/
│   └── js/
│
└── uploads/                        # User uploaded files (auto-generated)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (tested on 3.8, 3.9, 3.10, 3.11)
- **pip** (Python package manager)
- **Windows, macOS, or Linux**

### Automated Setup (Windows)

Run the automated setup script:

```batch
setup_and_run.bat
```

This will:
1. Check Python installation
2. Create virtual environment
3. Install dependencies
4. Optionally run all 3 phases automatically

### Manual Setup

1. **Clone or download this repository**

```bash
git clone https://github.com/AF8E/Prediction-Dashboard.git
cd Prediction-Dashboard
```

2. **Create virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🔧 Three-Phase Usage Guide

### Phase 1: Data Preparation (ETL)

Extract, transform, and merge your data:

```bash
python src/data_prep.py
```

**What it does:**
- Loads facility inventory from `data/generated_facility_data.xlsx`
- Loads work order history from `data/Simulated_Data.xlsx`
- **Extracts dates using regex patterns:**
  - `"Occurred: Feb 2025"` → `2025-02-01`
  - `"Occurred: February 15, 2025"` → `2025-02-15`
  - `"2025-02-15"` (ISO format) → `2025-02-15`
- **Extracts failure types:** compressor, electrical, HVAC, structural, etc.
- Generates synthetic failure history for facilities without records
- Merges all data into `data/Master_Training_Data.csv`

**Expected output:**
```
==============================================================
PHASE 1: DATA PREPARATION ETL
==============================================================

Loading facility inventory...
Loading work order history...
Extracting failure patterns from work orders...
Calculating failure statistics...
Generating synthetic failure history...
Merging datasets...

✓ Master training data saved to: data/Master_Training_Data.csv
  Total records: 248
  Total features: 25
```

---

### Phase 2: Model Training

Train machine learning models:

```bash
python src/train_model.py
```

**What it does:**
- Loads the master training data
- Prepares feature matrix with automatic encoding
- **Trains Random Forest Regressor** for condition prediction
- **Trains Random Forest Classifier** for failure probability
- Performs hyperparameter tuning with GridSearchCV (5-fold cross-validation)
- Saves models to `models/` directory

**Expected output:**
```
==============================================================
PHASE 2: MODEL TRAINING PIPELINE
==============================================================

Loading master training data...
Preparing features...

TRAINING CONDITION INDEX PREDICTOR (Regression)
Training set: 198 samples
Test set: 50 samples

Best parameters: {'max_depth': 20, 'min_samples_leaf': 1, ...}

Model Performance:
  Train RMSE: 3.24
  Test RMSE: 5.67
  Train R²: 0.9567
  Test R²: 0.8934

✓ Saved regression model: models/failure_model.pkl
✓ Saved classification model: models/classifier_model.pkl
✓ Saved scaler: models/scaler.pkl
✓ Saved label encoders: models/label_encoders.pkl
```

---

### Phase 3: Launch Dashboard

Start the Flask web application:

```bash
python app.py
```

**Access the dashboard:**
- Open your browser to: `http://localhost:5000`

**Available routes:**
- `/` - Dashboard home with summary statistics and Chart.js graph
- `/upload` - Upload new inventory files (drag & drop supported)
- `/predict?filename=your_file.xlsx` - View predictions
- `/api/predict` - REST API endpoint for programmatic access
- `/health` - Health check endpoint

---

## 📊 Using the Dashboard

### 1. Dashboard Home (`/`)

View overall facility health:
- **Total Facilities**: Number of facilities in inventory
- **High Risk Count**: Facilities with >70 risk score
- **Average Condition Index**: Fleet-wide health metric
- **Total Failures**: Recorded failure events
- **Degradation Chart**: Projected condition decline by facility age (Chart.js)

### 2. Upload New Data (`/upload`)

1. Click "Upload New Inventory"
2. Drag & drop or browse for your Excel/CSV file
3. Click "Run Prediction"

**Supported formats:**
- Excel: `.xlsx`, `.xls`
- CSV: `.csv`

**Required columns in upload file:**
- `Unique Identifier` - Facility ID (required)
- `Type` - Facility type (Foundation, HVAC, etc.) (required)
- `Condition Index` - Current condition (0-100) (required)
- `Age (years)` - Facility age (required)
- `Life Expectancy` - Expected lifespan (required)

**Optional columns:**
- `Title` - Facility name/description
- Any additional metadata

### 3. View Predictions (`/predict`)

Results table shows:
- **Predicted Condition Index**: ML-predicted current state
- **Failure Probability**: Likelihood of failure within 12 months (0-100%)
- **Predicted Failure Date**: Estimated failure date
- **Years Until Failure**: Time remaining
- **Risk Level**: High/Medium/Low classification with color-coded badges
- **Estimated Replacement Cost**: Cost estimate based on facility type

**Summary statistics:**
- Total facilities analyzed
- Risk distribution (High/Medium/Low counts)
- Average failure probability
- Total estimated replacement cost

**Print-friendly:** Results page includes print styling for reports

---

## 🧪 Testing

### Test with Sample Data

1. Use the provided Excel files in `data/` directory
2. Upload `generated_facility_data.xlsx` through the web interface
3. Verify predictions are generated

### API Testing

Test the prediction API endpoint:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Unique Identifier": ["FAC-001"],
    "Type": ["Foundation"],
    "Condition Index": [75],
    "Age (years)": [15],
    "Life Expectancy": [60]
  }'
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "Unique Identifier": "FAC-001",
      "Predicted_Condition_Index": 73.2,
      "Failure_Probability": 25.5,
      "Predicted_Failure_Date": "2030-05-15",
      "Years_Until_Failure": 4.3,
      "Estimated_Replacement_Cost": 550000,
      "Risk_Level": "Medium"
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2026-01-20T16:00:00"
}
```

---

## 🔍 How It Works

### Data Processing Pipeline

1. **Regex Pattern Extraction**
   - Extracts dates from text: `"Occurred: Feb 2025"` → `2025-02-01`
   - Identifies failure types: `"compressor failure"` → `compressor`
   - Handles multiple date formats (ISO, US, natural language)

2. **Feature Engineering**
   - Calculates `failure_rate` = failures / age
   - Computes `condition_delta` = 100 - condition
   - Generates `risk_score` from weighted features
   - Aggregates system-level statistics

3. **Synthetic Data Generation**
   - For facilities without failure history
   - Based on statistical distributions from real data
   - Weighted by age and condition

### Machine Learning Models

**Random Forest Regressor** (Condition Prediction)
- Predicts future condition index
- Features: age, life expectancy, failure rate, system conditions
- Hyperparameters tuned via 5-fold cross-validation
- Expected performance: R² > 0.85, RMSE < 8

**Random Forest Classifier** (Failure Probability)
- Predicts likelihood of failure within 12 months
- Handles class imbalance with weighted sampling
- Outputs probability scores (0-100%)
- Expected performance: Accuracy > 85%, F1 > 0.80

### Prediction Logic

```python
degradation_rate = (100 - predicted_condition) / age
years_until_critical = (predicted_condition - 25) / degradation_rate
years_until_failure = years_until_critical * (1 - failure_probability * 0.5)
predicted_failure_date = today + years_until_failure * 365 days
```

---

## 🛠️ Customization

### Adjust Failure Thresholds

Edit in `app.py`:

```python
# Change risk level thresholds
results['Risk_Level'] = pd.cut(
    failure_probability * 100,
    bins=[0, 20, 50, 100],  # Low: 0-20%, Med: 20-50%, High: 50-100%
    labels=['Low', 'Medium', 'High']
)
```

### Add New Failure Patterns

Edit `src/data_prep.py`:

```python
failure_patterns = {
    'your_pattern': r'your_regex_pattern',
    # Example: 'corrosion': r'(rust|corrosion|oxidation)',
}
```

### Modify Cost Estimates

Edit in `app.py`:

```python
base_replacement_cost = {
    'Foundation': 750000,  # Increase from 500000
    'HVAC': 300000,        # Increase from 200000
    # Add new facility types
    'Solar_Panel': 100000,
}
```

---

## 📈 Model Performance Metrics

Expected performance on test data:

| Metric | Regression Model | Classification Model |
|--------|------------------|----------------------|
| RMSE | 5-8 | N/A |
| R² Score | 0.85-0.95 | N/A |
| Accuracy | N/A | 85-92% |
| F1 Score | N/A | 0.78-0.88 |
| Precision | N/A | 75-85% |
| Recall | N/A | 80-90% |

---

## 🐛 Troubleshooting

### Models not loading

**Error:** `✗ Error loading models: [Errno 2] No such file or directory`

**Solution:** Run Phase 2 first:
```bash
python src/train_model.py
```

### CSV parsing errors

**Error:** `ParserError: Error tokenizing data`

**Solution:** Ensure your CSV uses UTF-8 encoding and standard delimiters

### Memory errors during training

**Error:** `MemoryError`

**Solution:** Reduce dataset size or use smaller hyperparameter grid in `src/train_model.py`:
```python
param_grid = {
    'n_estimators': [100],  # Reduce from [100, 200]
    'max_depth': [10]       # Reduce from [10, 20, None]
}
```

### File upload errors

**Error:** `413 Request Entity Too Large`

**Solution:** Increase `MAX_CONTENT_LENGTH` in `app.py` or reduce file size

### Port already in use

**Error:** `Address already in use`

**Solution:** Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

---

## 📝 Data Dictionary

### Master Training Data Columns

| Column | Type | Description |
|--------|------|-------------|
| `facility_id` | string | Unique facility identifier |
| `Type` | string | Facility type category |
| `Condition Index` | float | Current condition (0-100) |
| `Age (years)` | float | Facility age in years |
| `Life Expectancy` | float | Expected lifespan |
| `total_failures` | int | Number of recorded failures |
| `failure_rate` | float | Failures per year |
| `condition_delta` | float | Degradation from perfect |
| `risk_score` | float | Composite risk metric |
| `most_common_failure` | string | Primary failure type |
| `failure_within_12mo` | int | Binary target (0/1) |

---

## 🔐 Security Notes

**For Production Deployment:**

1. **Change the secret key** in `app.py`:
```python
import secrets
app.secret_key = secrets.token_hex(32)  # Generate secure random key
```

2. **File upload restrictions** (already implemented):
```python
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

3. **Enable HTTPS** and use environment variables for sensitive data

4. **Implement user authentication** if required

5. **Review DEPLOYMENT.md** for complete security checklist

---

## 📞 Support

For issues or questions:
- Create an issue on GitHub
- Review `DEPLOYMENT.md` for deployment guidance
- Check troubleshooting section above

---

## 📄 License

This project is proprietary software for US Space Command.

---

## 🎯 Roadmap

- [ ] Add automated retraining pipeline
- [ ] Implement real-time monitoring
- [ ] Add export to PDF reports
- [ ] Integrate with CMMS systems
- [ ] Add multi-site comparison dashboard
- [ ] Implement role-based access control
- [ ] Add data visualization enhancements
- [ ] Support for additional file formats

---

## 🏗️ Architecture

### Technology Stack

- **Backend:** Flask 3.0 (Python web framework)
- **Machine Learning:** scikit-learn 1.3.0 (Random Forest)
- **Data Processing:** pandas 2.1.0, numpy 1.24.3
- **Frontend:** Bootstrap 5.3.0, Chart.js 4.3.0 (CDN)
- **File Handling:** openpyxl, xlrd (Excel support)

### Design Patterns

- **ETL Pipeline:** Extract-Transform-Load for data preparation
- **ML Pipeline:** Train-Validate-Test workflow
- **MVC Architecture:** Flask routes (controllers), templates (views), models (data)
- **RESTful API:** JSON endpoints for programmatic access

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Maintained by:** AF8E  
**Python Compatibility:** 3.8+

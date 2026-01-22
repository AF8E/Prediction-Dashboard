"""
Phase 2: Model Training Script
Trains Random Forest models for condition prediction and failure classification

This script:
1. Loads master training data
2. Prepares features with encoding and scaling
3. Trains Random Forest Regressor for condition prediction
4. Trains Random Forest Classifier for failure probability
5. Performs hyperparameter tuning with GridSearchCV
6. Saves models and preprocessing objects
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class FacilityMLPipeline:
    def __init__(self, data_path):
        """
        Initialize ML pipeline
        
        Args:
            data_path: Path to Master_Training_Data.csv
        """
        self.data_path = data_path
        self.data = None
        self.regression_model = None
        self.classification_model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load master training data"""
        print("Loading master training data...")
        self.data = pd.read_csv(self.data_path)
        print(f"  Loaded {len(self.data)} records with {len(self.data.columns)} features")
        return self.data
    
    def prepare_features(self):
        """
        Prepare feature matrix and target variables
        
        Returns:
            X: Feature matrix (scaled)
            y_condition: Target for condition prediction
            y_failure: Target for failure classification
        """
        print("\nPreparing features...")
        
        df = self.data.copy()
        
        # Define feature columns (numeric and categorical)
        numeric_features = [
            'Age (years)',
            'Life Expectancy', 
            'total_failures',
            'failure_rate',
            'condition_delta',
            'risk_score',
            'days_between_failures'
        ]
        
        # Add system-level features if available
        if 'avg_system_condition' in df.columns:
            numeric_features.extend([
                'avg_system_condition',
                'min_system_condition', 
                'system_condition_std',
                'avg_system_life_expectancy'
            ])
        
        categorical_features = [
            'most_common_failure'
        ]
        
        # Add facility type if available
        if 'Type' in df.columns:
            categorical_features.append('Type')
        if 'Title' in df.columns:
            categorical_features.append('Title')
        
        # Filter features that exist in dataframe
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Handle missing values
        for col in numeric_features:
            df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_features:
            df[col] = df[col].fillna('Unknown')
        
        # Encode categorical features
        for col in categorical_features:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Combine features
        encoded_categorical = [f'{col}_encoded' for col in categorical_features]
        all_features = numeric_features + encoded_categorical
        
        X = df[all_features].copy()
        
        # Target variables
        y_condition = df['Condition Index'] if 'Condition Index' in df.columns else df['risk_score']
        y_failure = df['failure_within_12mo'] if 'failure_within_12mo' in df.columns else (df['total_failures'] > 0).astype(int)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"  Feature matrix: {X_scaled.shape}")
        print(f"  Features: {list(X_scaled.columns)}")
        
        return X_scaled, y_condition, y_failure
    
    def train_condition_predictor(self, X, y):
        """
        Train Random Forest Regressor for Condition Index prediction
        
        Uses GridSearchCV for hyperparameter tuning with 5-fold cross-validation
        
        Args:
            X: Feature matrix
            y: Target condition index
            
        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("TRAINING CONDITION INDEX PREDICTOR (Regression)")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Define model with hyperparameter search
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        print("\nPerforming hyperparameter search...")
        grid_search = GridSearchCV(
            rf_regressor,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Get best model
        self.regression_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred_train = self.regression_model.predict(X_train)
        y_pred_test = self.regression_model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("\nModel Performance:")
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.regression_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.feature_importance['regression'] = feature_importance
        
        return self.regression_model
    
    def train_failure_classifier(self, X, y):
        """
        Train Random Forest Classifier for failure probability
        
        Handles class imbalance with balanced class weights
        
        Args:
            X: Feature matrix
            y: Target failure indicator
            
        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("TRAINING FAILURE PROBABILITY CLASSIFIER")
        print("="*60)
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"\nClass distribution:")
        print(f"  No failure (0): {class_counts.get(0, 0)}")
        print(f"  Failure (1): {class_counts.get(1, 0)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        class_weight = 'balanced' if class_counts.min() / class_counts.max() < 0.3 else None
        
        # Define model
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_classifier = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight
        )
        
        print("\nPerforming hyperparameter search...")
        grid_search = GridSearchCV(
            rf_classifier,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Get best model
        self.classification_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred_test = self.classification_model.predict(X_test)
        y_pred_proba = self.classification_model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['No Failure', 'Failure']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.classification_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.feature_importance['classification'] = feature_importance
        
        return self.classification_model
    
    def save_models(self, output_dir='models'):
        """
        Save trained models and preprocessing objects
        
        Saves:
        - failure_model.pkl (regression model)
        - classifier_model.pkl (classification model)
        - scaler.pkl (feature scaler)
        - label_encoders.pkl (categorical encoders)
        - model_metadata.json (training metadata)
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Save models
        joblib.dump(self.regression_model, output_path / 'failure_model.pkl')
        print(f"✓ Saved regression model: {output_path / 'failure_model.pkl'}")
        
        joblib.dump(self.classification_model, output_path / 'classifier_model.pkl')
        print(f"✓ Saved classification model: {output_path / 'classifier_model.pkl'}")
        
        # Save scaler
        joblib.dump(self.scaler, output_path / 'scaler.pkl')
        print(f"✓ Saved scaler: {output_path / 'scaler.pkl'}")
        
        # Save label encoders
        joblib.dump(self.label_encoders, output_path / 'label_encoders.pkl')
        print(f"✓ Saved label encoders: {output_path / 'label_encoders.pkl'}")
        
        # Save feature importance
        for model_type, importance_df in self.feature_importance.items():
            importance_df.to_csv(output_path / f'{model_type}_feature_importance.csv', index=False)
            print(f"✓ Saved {model_type} feature importance")
        
        # Save metadata
        metadata = {
            'training_date': pd.Timestamp.now().isoformat(),
            'training_samples': len(self.data),
            'regression_model_type': str(type(self.regression_model).__name__),
            'classification_model_type': str(type(self.classification_model).__name__),
            'features': list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else []
        }
        
        with open(output_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {output_path / 'model_metadata.json'}")
        
        return output_path
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        print("="*60)
        print("PHASE 2: MODEL TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        self.load_data()
        
        # Prepare features
        X, y_condition, y_failure = self.prepare_features()
        
        # Train models
        self.train_condition_predictor(X, y_condition)
        self.train_failure_classifier(X, y_failure)
        
        # Save models
        model_path = self.save_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nModels saved to: {model_path}")
        print("\nNext step: Run Phase 3 to create the Flask dashboard!")
        
        return model_path


# Main execution
if __name__ == "__main__":
    # Path to master training data
    DATA_FILE = "data/Master_Training_Data.csv"
    
    # Check if data exists
    if not Path(DATA_FILE).exists():
        print(f"ERROR: {DATA_FILE} not found!")
        print("Please run Phase 1 (src/data_prep.py) first.")
        exit(1)
    
    # Run training pipeline
    pipeline = FacilityMLPipeline(DATA_FILE)
    model_path = pipeline.run_training_pipeline()

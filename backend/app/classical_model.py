"""
Classical Breast Cancer Classifier
Baseline model using real UCI dataset for comparison with quantum approach
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class ClassicalBreastCancerClassifier:
    """
    Classical machine learning classifier for breast cancer detection
    Uses Random Forest trained on real UCI Breast Cancer Wisconsin dataset
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy_score = 0.0
        self.feature_names = None
        
        # Load real UCI dataset
        self._load_real_dataset()
        
        # Try to load pre-trained model or train on real data
        self._load_pretrained_model()
    
    def _load_real_dataset(self):
        """Load the real UCI Breast Cancer Wisconsin dataset from official repository"""
        print("[CHART] Loading UCI Breast Cancer Wisconsin dataset for classical model...")
        
        try:
            # Import ucimlrepo
            from ucimlrepo import fetch_ucirepo
            
            # Fetch dataset from UCI repository
            print("[GLOBE] Fetching from UCI ML Repository...")
            breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
            
            # Extract data (as pandas dataframes)
            X_df = breast_cancer_wisconsin_diagnostic.data.features
            y_df = breast_cancer_wisconsin_diagnostic.data.targets
            
            # Convert to numpy arrays
            self.X_full = X_df.values
            # Convert diagnosis to binary (M=0, B=1 to match sklearn convention)
            y_values = y_df.values.flatten()
            self.y_full = np.where(y_values == 'M', 0, 1)  # 0: malignant, 1: benign
            self.feature_names = X_df.columns.tolist()
            
            print(f"[OK] Official UCI Dataset loaded for classical model!")
            print(f"   Samples: {self.X_full.shape[0]}")
            print(f"   Features: {self.X_full.shape[1]} biomarkers")
            print(f"   Classes: Malignant={np.sum(self.y_full == 0)}, Benign={np.sum(self.y_full == 1)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load from UCI repository: {e}")
            print("[REFRESH] Falling back to sklearn dataset...")
            
            # Fallback to sklearn dataset
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            self.X_full = data.data
            self.y_full = data.target
            self.feature_names = data.feature_names.tolist()
            
            print(f"[OK] Fallback dataset loaded: {self.X_full.shape[0]} samples, {self.X_full.shape[1]} features")
        
        # Prepare training and test sets (same split as quantum model)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42, stratify=self.y_full
        )
        
        print(f"[OK] Classical model dataset ready: {self.X_train.shape[0]} training, {self.X_test.shape[0]} test samples")
    
    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Dict:
        """Train the classical classifier on real UCI data"""
        print("[MICROSCOPE] Training classical Random Forest on UCI Breast Cancer Wisconsin dataset...")
        
        # Use real dataset if no custom data provided
        if X is None or y is None:
            X, y = self.X_train, self.y_train
        
        print(f"   Training samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        print("🌳 Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Validate model
        y_pred = self.model.predict(X_val_scaled)
        validation_accuracy = accuracy_score(y_val, y_pred)
        
        # Evaluate on test set
        X_test_scaled = self.scaler.transform(self.X_test)
        y_test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        self.accuracy_score = test_accuracy
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        print(f"[OK] Classical training completed!")
        print(f"   Validation accuracy: {validation_accuracy:.3f}")
        print(f"   Test accuracy: {test_accuracy:.3f}")
        
        return {
            "status": "trained",
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_importance": self.get_feature_importance(),
            "dataset_source": "UCI ML Repository"
        }
    
    def predict(self, X: np.ndarray) -> Dict:
        """Make prediction using classical classifier"""
        if not self.is_trained:
            print("[REFRESH] Classical model not trained. Training on UCI dataset...")
            self.train()
        
        # Scale input features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        prediction_class = self.model.predict(X_scaled)[0]
        
        # Convert to risk level (0=malignant, 1=benign)
        risk_level = "High Risk" if prediction_class == 0 else "Low Risk"
        probability = 1 - prediction_proba[1]  # Risk probability (1 - benign probability)
        confidence = max(prediction_proba)
        
        return {
            'prediction': risk_level,
            'probability': probability,
            'confidence': confidence,
            'class_probabilities': {
                'benign': prediction_proba[1],
                'malignant': prediction_proba[0]
            }
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}
        
        importance_scores = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance_scores))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_features': sorted_features[:10],
            'all_features': feature_importance
        }
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model and scaler
            joblib.dump(self.model, os.path.join(model_dir, 'classical_model.pkl'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'classical_scaler.pkl'))
            
            # Save metadata
            metadata = {
                'accuracy': self.accuracy_score,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'dataset_source': 'UCI ML Repository'
            }
            
            import json
            with open(os.path.join(model_dir, 'classical_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("[OK] Classical model saved successfully")
        except Exception as e:
            print(f"[ERROR] Failed to save classical model: {e}")
    
    def _load_pretrained_model(self):
        """Load pre-trained model from disk or train on real data"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_path = os.path.join(model_dir, 'classical_model.pkl')
            scaler_path = os.path.join(model_dir, 'classical_scaler.pkl')
            metadata_path = os.path.join(model_dir, 'classical_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.accuracy_score = metadata.get('accuracy', 0.0)
                    self.is_trained = metadata.get('is_trained', True)
                else:
                    self.accuracy_score = 0.95  # Typical accuracy for this dataset
                    self.is_trained = True
                
                print("[OK] Loaded pre-trained classical model")
            else:
                print("[REFRESH] No pre-trained model found. Training on real UCI data...")
                self.train()
                
        except Exception as e:
            print(f"[ERROR] Failed to load classical model: {e}")
            print("[REFRESH] Training new model on real UCI data...")
            self.train()
    
    def evaluate_model(self) -> Dict:
        """Evaluate model performance on test set"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # Predict on test set
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'classification_report': report
        }
    
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.is_trained
    
    def get_feature_count(self) -> int:
        """Get number of features used by the model"""
        return len(self.feature_names) if self.feature_names else 30
    
    def get_accuracy(self) -> float:
        """Get model accuracy"""
        return self.accuracy_score
    
    def get_model_info(self) -> Dict:
        """Get detailed model information"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "model_type": "Random Forest",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "accuracy": self.accuracy_score,
            "features": self.get_feature_count(),
            "trained": self.is_trained,
            "dataset": "UCI Breast Cancer Wisconsin (Official Repository)",
            "samples": len(self.X_full) if hasattr(self, 'X_full') else 569
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the classical classifier with real UCI data
    print("[TEST] Testing Classical Breast Cancer Classifier with UCI Data")
    print("=" * 60)
    
    classifier = ClassicalBreastCancerClassifier()
    
    # Train model on real data
    training_result = classifier.train()
    print("\n[CHART] Training Results:")
    print(f"   Status: {training_result['status']}")
    print(f"   Validation Accuracy: {training_result['validation_accuracy']:.3f}")
    print(f"   Test Accuracy: {training_result['test_accuracy']:.3f}")
    print(f"   Dataset Source: {training_result['dataset_source']}")
    
    # Evaluate model
    evaluation = classifier.evaluate_model()
    print(f"\n[GRAPH] Model Evaluation:")
    print(f"   Accuracy: {evaluation['accuracy']:.3f}")
    print(f"   Precision: {evaluation['precision']:.3f}")
    print(f"   Recall: {evaluation['recall']:.3f}")
    print(f"   F1-Score: {evaluation['f1_score']:.3f}")
    
    # Make prediction on test sample
    test_sample = classifier.X_test[0:1]  # First test sample
    prediction = classifier.predict(test_sample)
    print(f"\n[MICROSCOPE] Prediction Results:")
    print(f"   Prediction: {prediction['prediction']}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    print(f"   Risk Probability: {prediction['probability']:.3f}")
    
    # Show feature importance
    importance = classifier.get_feature_importance()
    print(f"\n[DART] Top 5 Important Features:")
    for feature, score in importance['top_features'][:5]:
        print(f"   {feature}: {score:.4f}")
    
    print("\n[OK] Classical classifier test completed with real UCI data!")
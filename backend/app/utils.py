"""
Utility functions for Q-MediScan backend
Data preprocessing, validation, and helper functions using real UCI data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the real UCI Breast Cancer Wisconsin dataset from official repository
    Returns: X (features), y (labels), feature_names
    """
    try:
        # Import ucimlrepo
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset from UCI repository
        print("[GLOBE] Fetching UCI Breast Cancer Wisconsin dataset from official repository...")
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
        
        # Extract data (as pandas dataframes)
        X_df = breast_cancer_wisconsin_diagnostic.data.features
        y_df = breast_cancer_wisconsin_diagnostic.data.targets
        
        # Convert to numpy arrays
        X = X_df.values
        # Convert diagnosis to binary (M=0, B=1 to match sklearn convention)
        y_values = y_df.values.flatten()
        y = np.where(y_values == 'M', 0, 1)  # 0: malignant, 1: benign
        feature_names = X_df.columns.tolist()
        
        print(f"[OK] Official UCI Breast Cancer Wisconsin dataset loaded!")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]} biomarkers")
        print(f"   Target distribution: Malignant={np.sum(y==0)}, Benign={np.sum(y==1)}")
        print(f"   Repository: {breast_cancer_wisconsin_diagnostic.metadata.repository_url}")
        
        return X, y, feature_names
        
    except Exception as e:
        print(f"[ERROR] Failed to load from UCI repository: {e}")
        print("[REFRESH] Falling back to sklearn dataset...")
        
        # Fallback to sklearn dataset
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names.tolist()
        
        print(f"[OK] Fallback dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names

def preprocess_data(X: np.ndarray, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """
    Preprocess input data for model prediction
    
    Args:
        X: Input features array
        scaler: Pre-fitted scaler (optional)
    
    Returns:
        Preprocessed features
    """
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure correct shape
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Validate feature count
    expected_features = 30
    if X.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
    
    # Scale features if scaler provided
    if scaler is not None:
        X = scaler.transform(X)
    else:
        # Apply standard normalization
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    return X

def validate_input_data(data: Dict) -> bool:
    """
    Validate input patient data
    
    Args:
        data: Dictionary containing patient biomarker data
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
        'mean_smoothness', 'mean_compactness', 'mean_concavity',
        'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'radius_error', 'texture_error', 'perimeter_error', 'area_error',
        'smoothness_error', 'compactness_error', 'concavity_error',
        'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
        'worst_smoothness', 'worst_compactness', 'worst_concavity',
        'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
    ]
    
    # Check all required fields are present
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Check all values are numeric
    for field in required_fields:
        try:
            float(data[field])
        except (ValueError, TypeError):
            raise ValueError(f"Field '{field}' must be a numeric value")
    
    # Check for reasonable value ranges (basic sanity checks)
    if data['mean_radius'] < 0 or data['mean_radius'] > 50:
        raise ValueError("Mean radius must be between 0 and 50")
    
    if data['mean_area'] < 0 or data['mean_area'] > 5000:
        raise ValueError("Mean area must be between 0 and 5000")
    
    return True

def generate_sample_data() -> Dict:
    """
    Generate realistic sample patient data from real UCI dataset
    
    Returns:
        Dictionary with sample biomarker values based on real data
    """
    # Load real dataset to get realistic sample
    try:
        X, y, feature_names = load_breast_cancer_dataset()
        
        # Get random sample from real dataset
        sample_idx = np.random.randint(0, len(X))
        sample_values = X[sample_idx]
        actual_diagnosis = 'Malignant' if y[sample_idx] == 0 else 'Benign'
        
        # Create dictionary with feature names mapped to API format
        feature_mapping = {
            'mean radius': 'mean_radius',
            'mean texture': 'mean_texture', 
            'mean perimeter': 'mean_perimeter',
            'mean area': 'mean_area',
            'mean smoothness': 'mean_smoothness',
            'mean compactness': 'mean_compactness',
            'mean concavity': 'mean_concavity',
            'mean concave points': 'mean_concave_points',
            'mean symmetry': 'mean_symmetry',
            'mean fractal dimension': 'mean_fractal_dimension',
            'radius error': 'radius_error',
            'texture error': 'texture_error',
            'perimeter error': 'perimeter_error',
            'area error': 'area_error',
            'smoothness error': 'smoothness_error',
            'compactness error': 'compactness_error',
            'concavity error': 'concavity_error',
            'concave points error': 'concave_points_error',
            'symmetry error': 'symmetry_error',
            'fractal dimension error': 'fractal_dimension_error',
            'worst radius': 'worst_radius',
            'worst texture': 'worst_texture',
            'worst perimeter': 'worst_perimeter',
            'worst area': 'worst_area',
            'worst smoothness': 'worst_smoothness',
            'worst compactness': 'worst_compactness',
            'worst concavity': 'worst_concavity',
            'worst concave points': 'worst_concave_points',
            'worst symmetry': 'worst_symmetry',
            'worst fractal dimension': 'worst_fractal_dimension'
        }
        
        sample_data = {}
        for i, feature_name in enumerate(feature_names):
            api_name = feature_mapping.get(feature_name, feature_name.replace(' ', '_'))
            sample_data[api_name] = float(sample_values[i])
        
        # Add metadata
        sample_data['_metadata'] = {
            'source': 'UCI ML Repository',
            'sample_index': int(sample_idx),
            'actual_diagnosis': actual_diagnosis
        }
        
        return sample_data
        
    except Exception as e:
        print(f"[WARNING]  Could not load real data, using synthetic sample: {e}")
        # Fallback to synthetic data based on known ranges
        sample_data = {
            'mean_radius': np.random.uniform(6, 28),
            'mean_texture': np.random.uniform(9, 40),
            'mean_perimeter': np.random.uniform(43, 189),
            'mean_area': np.random.uniform(143, 2501),
            'mean_smoothness': np.random.uniform(0.05, 0.16),
            'mean_compactness': np.random.uniform(0.02, 0.35),
            'mean_concavity': np.random.uniform(0, 0.43),
            'mean_concave_points': np.random.uniform(0, 0.20),
            'mean_symmetry': np.random.uniform(0.11, 0.30),
            'mean_fractal_dimension': np.random.uniform(0.05, 0.10),
            'radius_error': np.random.uniform(0.11, 2.87),
            'texture_error': np.random.uniform(0.36, 4.88),
            'perimeter_error': np.random.uniform(0.76, 21.98),
            'area_error': np.random.uniform(6.8, 542.2),
            'smoothness_error': np.random.uniform(0.002, 0.031),
            'compactness_error': np.random.uniform(0.002, 0.135),
            'concavity_error': np.random.uniform(0, 0.396),
            'concave_points_error': np.random.uniform(0, 0.053),
            'symmetry_error': np.random.uniform(0.008, 0.079),
            'fractal_dimension_error': np.random.uniform(0.001, 0.030),
            'worst_radius': np.random.uniform(7.9, 36.0),
            'worst_texture': np.random.uniform(12.0, 49.5),
            'worst_perimeter': np.random.uniform(50.4, 251.2),
            'worst_area': np.random.uniform(185.2, 4254.0),
            'worst_smoothness': np.random.uniform(0.071, 0.223),
            'worst_compactness': np.random.uniform(0.027, 1.058),
            'worst_concavity': np.random.uniform(0, 1.252),
            'worst_concave_points': np.random.uniform(0, 0.291),
            'worst_symmetry': np.random.uniform(0.156, 0.664),
            'worst_fractal_dimension': np.random.uniform(0.055, 0.208),
            '_metadata': {
                'source': 'Synthetic',
                'actual_diagnosis': 'Unknown'
            }
        }
        
        return sample_data

def create_sample_csv(filename: str = "sample_patient_data.csv", n_samples: int = 5):
    """
    Create a sample CSV file with real patient data from UCI repository
    
    Args:
        filename: Output CSV filename
        n_samples: Number of sample patients to generate
    """
    try:
        # Load real dataset
        X, y, feature_names = load_breast_cancer_dataset()
        
        # Select random samples from real data
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        # Create DataFrame with real samples
        samples = []
        for i, idx in enumerate(sample_indices):
            sample = {}
            sample['patient_id'] = f"UCI_PATIENT_{idx:03d}"
            sample['actual_diagnosis'] = 'Malignant' if y[idx] == 0 else 'Benign'
            sample['data_source'] = 'UCI ML Repository'
            
            # Map feature names to API format
            feature_mapping = {
                'mean radius': 'mean_radius',
                'mean texture': 'mean_texture', 
                'mean perimeter': 'mean_perimeter',
                'mean area': 'mean_area',
                'mean smoothness': 'mean_smoothness',
                'mean compactness': 'mean_compactness',
                'mean concavity': 'mean_concavity',
                'mean concave points': 'mean_concave_points',
                'mean symmetry': 'mean_symmetry',
                'mean fractal dimension': 'mean_fractal_dimension',
                'radius error': 'radius_error',
                'texture error': 'texture_error',
                'perimeter error': 'perimeter_error',
                'area error': 'area_error',
                'smoothness error': 'smoothness_error',
                'compactness error': 'compactness_error',
                'concavity error': 'concavity_error',
                'concave points error': 'concave_points_error',
                'symmetry error': 'symmetry_error',
                'fractal dimension error': 'fractal_dimension_error',
                'worst radius': 'worst_radius',
                'worst texture': 'worst_texture',
                'worst perimeter': 'worst_perimeter',
                'worst area': 'worst_area',
                'worst smoothness': 'worst_smoothness',
                'worst compactness': 'worst_compactness',
                'worst concavity': 'worst_concavity',
                'worst concave points': 'worst_concave_points',
                'worst symmetry': 'worst_symmetry',
                'worst fractal dimension': 'worst_fractal_dimension'
            }
            
            for j, feature_name in enumerate(feature_names):
                api_name = feature_mapping.get(feature_name, feature_name.replace(' ', '_'))
                sample[api_name] = X[idx, j]
            
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        # Reorder columns to put metadata first
        metadata_cols = ['patient_id', 'actual_diagnosis', 'data_source']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        df = df[metadata_cols + feature_cols]
        
        # Save to data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[OK] Created sample CSV with {n_samples} real UCI patients: {filepath}")
        print(f"   Diagnoses: {df['actual_diagnosis'].value_counts().to_dict()}")
        print(f"   Data Source: UCI ML Repository")
        return filepath
        
    except Exception as e:
        print(f"[ERROR] Failed to create real sample CSV: {e}")
        # Fallback to synthetic data
        samples = []
        for i in range(n_samples):
            sample = generate_sample_data()
            sample['patient_id'] = f"SYNTHETIC_{i+1:03d}"
            if '_metadata' in sample:
                sample['actual_diagnosis'] = sample['_metadata']['actual_diagnosis']
                sample['data_source'] = sample['_metadata']['source']
                del sample['_metadata']
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        # Save to data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[WARNING]  Created synthetic sample CSV: {filepath}")
        return filepath

def calculate_risk_score(prediction_result: Dict) -> Dict:
    """
    Calculate comprehensive risk score from prediction results
    
    Args:
        prediction_result: Result from quantum or classical model
    
    Returns:
        Enhanced risk assessment
    """
    probability = prediction_result.get('probability', 0.5)
    confidence = prediction_result.get('confidence', 0.5)
    
    # Calculate risk score (0-100)
    risk_score = int(probability * 100)
    
    # Determine risk category
    if risk_score < 20:
        risk_category = "Very Low"
        color = "green"
    elif risk_score < 40:
        risk_category = "Low"
        color = "lightgreen"
    elif risk_score < 60:
        risk_category = "Moderate"
        color = "yellow"
    elif risk_score < 80:
        risk_category = "High"
        color = "orange"
    else:
        risk_category = "Very High"
        color = "red"
    
    # Generate recommendations
    recommendations = generate_recommendations(risk_score, confidence)
    
    return {
        'risk_score': risk_score,
        'risk_category': risk_category,
        'color': color,
        'confidence_level': confidence,
        'recommendations': recommendations
    }

def generate_recommendations(risk_score: int, confidence: float) -> List[str]:
    """
    Generate personalized recommendations based on risk assessment
    
    Args:
        risk_score: Risk score (0-100)
        confidence: Model confidence (0-1)
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if risk_score < 20:
        recommendations.extend([
            "Continue regular annual screenings",
            "Maintain healthy lifestyle habits",
            "Monitor for any changes in breast tissue"
        ])
    elif risk_score < 40:
        recommendations.extend([
            "Consider more frequent screenings (every 6-12 months)",
            "Discuss family history with your doctor",
            "Maintain healthy diet and exercise routine"
        ])
    elif risk_score < 60:
        recommendations.extend([
            "Schedule immediate consultation with oncologist",
            "Consider additional imaging studies",
            "Discuss genetic testing options"
        ])
    elif risk_score < 80:
        recommendations.extend([
            "Urgent medical evaluation recommended",
            "Consider MRI or ultrasound imaging",
            "Discuss biopsy options with specialist"
        ])
    else:
        recommendations.extend([
            "Immediate medical attention required",
            "Schedule urgent oncology consultation",
            "Consider comprehensive diagnostic workup"
        ])
    
    # Add confidence-based recommendations
    if confidence < 0.7:
        recommendations.append("Consider retesting with additional biomarkers")
    
    # Always add disclaimer
    recommendations.append("Remember: This is an AI assessment tool, not a medical diagnosis")
    
    return recommendations

def format_biomarker_report(patient_data: Dict, prediction_result: Dict) -> str:
    """
    Generate a formatted biomarker analysis report
    
    Args:
        patient_data: Patient biomarker data
        prediction_result: Model prediction results
    
    Returns:
        Formatted report string
    """
    risk_assessment = calculate_risk_score(prediction_result)
    
    report = f"""
=== Q-MediScan Biomarker Analysis Report ===

RISK ASSESSMENT:
- Risk Level: {prediction_result['prediction']}
- Risk Score: {risk_assessment['risk_score']}/100
- Confidence: {prediction_result['confidence']:.1%}
- Category: {risk_assessment['risk_category']}

KEY BIOMARKERS:
- Mean Radius: {patient_data['mean_radius']:.2f}
- Mean Texture: {patient_data['mean_texture']:.2f}
- Mean Area: {patient_data['mean_area']:.2f}
- Worst Radius: {patient_data['worst_radius']:.2f}

RECOMMENDATIONS:
"""
    
    for i, rec in enumerate(risk_assessment['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += f"""
TECHNICAL DETAILS:
- Analysis Method: Quantum Machine Learning
- Model Confidence: {prediction_result['confidence']:.1%}
- Dataset: UCI Breast Cancer Wisconsin (Official Repository)
- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER:
This analysis is for research and educational purposes only.
Always consult qualified healthcare professionals for medical decisions.
"""
    
    return report

def get_dataset_statistics() -> Dict:
    """
    Get statistics about the UCI Breast Cancer Wisconsin dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    try:
        X, y, feature_names = load_breast_cancer_dataset()
        
        stats = {
            'total_samples': len(X),
            'total_features': len(feature_names),
            'malignant_cases': int(np.sum(y == 0)),
            'benign_cases': int(np.sum(y == 1)),
            'feature_statistics': {},
            'feature_names': feature_names,
            'data_source': 'UCI ML Repository'
        }
        
        # Calculate feature statistics
        for i, feature_name in enumerate(feature_names):
            stats['feature_statistics'][feature_name] = {
                'mean': float(np.mean(X[:, i])),
                'std': float(np.std(X[:, i])),
                'min': float(np.min(X[:, i])),
                'max': float(np.max(X[:, i])),
                'median': float(np.median(X[:, i]))
            }
        
        return stats
        
    except Exception as e:
        print(f"[ERROR] Failed to get dataset statistics: {e}")
        return {}

# Example usage
if __name__ == "__main__":
    # Test utility functions with real UCI data
    print("[TEST] Testing Q-MediScan utilities with real UCI data...")
    print("=" * 60)
    
    # Load dataset
    X, y, features = load_breast_cancer_dataset()
    print(f"[OK] Dataset loaded: {X.shape}")
    
    # Generate sample data from real dataset
    sample = generate_sample_data()
    print("[OK] Real sample data generated")
    if '_metadata' in sample:
        print(f"   Source: {sample['_metadata']['source']}")
        print(f"   Actual diagnosis: {sample['_metadata']['actual_diagnosis']}")
    
    # Validate sample data
    try:
        # Remove metadata for validation
        sample_for_validation = {k: v for k, v in sample.items() if not k.startswith('_')}
        validate_input_data(sample_for_validation)
        print("[OK] Sample data validation: PASSED")
    except ValueError as e:
        print(f"[ERROR] Sample data validation: FAILED - {e}")
    
    # Create sample CSV with real data
    csv_path = create_sample_csv("test_real_uci_patients.csv", 3)
    print(f"[OK] Real UCI sample CSV created: {csv_path}")
    
    # Test preprocessing
    X_sample = np.array([v for k, v in sample.items() if not k.startswith('_')]).reshape(1, -1)
    X_processed = preprocess_data(X_sample)
    print(f"[OK] Preprocessing test: {X_processed.shape}")
    
    # Get dataset statistics
    stats = get_dataset_statistics()
    print(f"[OK] Dataset statistics: {stats['total_samples']} samples, {stats['total_features']} features")
    print(f"   Data source: {stats['data_source']}")
    
    print("\n[PARTY] All utility tests completed with real UCI data!")
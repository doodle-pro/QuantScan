"""
Advanced Quantum Benchmarking and Validation System
Comprehensive testing framework for quantum medical AI systems
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

class QuantumMedicalBenchmark:
    """
    Comprehensive benchmarking system for quantum medical AI
    Validates quantum advantage and ensures medical-grade reliability
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.benchmark_results = {}
        self.statistical_tests = {}
        self.medical_validation = {}
        
        # Medical AI requirements
        self.medical_thresholds = {
            'minimum_accuracy': 0.85,      # 85% minimum for medical AI
            'minimum_sensitivity': 0.90,   # 90% sensitivity for cancer detection
            'minimum_specificity': 0.80,   # 80% specificity to avoid false positives
            'maximum_false_negative': 0.10, # Max 10% false negatives
            'reliability_threshold': 0.95   # 95% reliability across tests
        }
    
    def benchmark_quantum_vs_classical(self, quantum_model, X_train: np.ndarray, 
                                     X_test: np.ndarray, y_train: np.ndarray, 
                                     y_test: np.ndarray) -> Dict:
        """
        Comprehensive benchmark comparing quantum vs classical models
        Tests multiple classical baselines against quantum implementation
        """
        print("[MICROSCOPE] Starting comprehensive quantum vs classical benchmark...")
        
        results = {
            'quantum_results': {},
            'classical_results': {},
            'comparison': {},
            'statistical_significance': {},
            'medical_validation': {}
        }
        
        # Benchmark quantum model
        print("[ATOM]  Benchmarking quantum model...")
        quantum_results = self._benchmark_single_model(
            quantum_model, X_train, X_test, y_train, y_test, "Quantum ML"
        )
        results['quantum_results'] = quantum_results
        
        # Benchmark classical models
        classical_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        for name, model in classical_models.items():
            print(f"🖥️  Benchmarking {name}...")
            classical_result = self._benchmark_single_model(
                model, X_train, X_test, y_train, y_test, name
            )
            results['classical_results'][name] = classical_result
        
        # Statistical comparison
        results['comparison'] = self._compare_models(quantum_results, results['classical_results'])
        
        # Statistical significance testing
        results['statistical_significance'] = self._test_statistical_significance(
            quantum_results, results['classical_results']
        )
        
        # Medical validation
        results['medical_validation'] = self._validate_medical_requirements(
            quantum_results, results['classical_results']
        )
        
        # Store results
        self.benchmark_results = results
        
        return results
    
    def _benchmark_single_model(self, model, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray, 
                               model_name: str) -> Dict:
        """Comprehensive benchmarking of a single model"""
        
        # Training phase
        start_time = time.time()
        
        if hasattr(model, 'train'):
            # Quantum model
            training_result = model.train(X_train, y_train)
            training_time = time.time() - start_time
        else:
            # Classical model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_result = {'status': 'trained'}
        
        # Prediction phase
        start_time = time.time()
        
        if hasattr(model, 'predict') and hasattr(model, 'is_trained'):
            # Quantum model - predict each sample individually
            y_pred = []
            y_proba = []
            
            for i in range(len(X_test)):
                pred_result = model.predict(X_test[i:i+1])
                prediction = 1 if pred_result['prediction'] == 'Low Risk' else 0
                probability = pred_result.get('probability', 0.5)
                
                y_pred.append(prediction)
                y_proba.append([1-probability, probability])
            
            y_pred = np.array(y_pred)
            y_proba = np.array(y_proba)
            
        else:
            # Classical model
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            else:
                y_proba = np.column_stack([1-y_pred, y_pred])
        
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_proba)
        
        # Cross-validation (for classical models)
        cv_scores = None
        if not hasattr(model, 'train'):  # Classical model
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            except Exception as e:
                warnings.warn(f"Cross-validation failed for {model_name}: {e}")
                cv_scores = None
        
        return {
            'model_name': model_name,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'training_result': training_result,
            'metrics': metrics,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            roc_auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Medical-specific metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Error rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Medical decision metrics
        diagnostic_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'diagnostic_odds_ratio': diagnostic_odds_ratio,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def _compare_models(self, quantum_results: Dict, classical_results: Dict) -> Dict:
        """Compare quantum model against classical baselines"""
        
        comparison = {
            'quantum_advantages': {},
            'classical_advantages': {},
            'performance_summary': {},
            'efficiency_analysis': {}
        }
        
        quantum_metrics = quantum_results['metrics']
        
        # Compare against each classical model
        for classical_name, classical_result in classical_results.items():
            classical_metrics = classical_result['metrics']
            
            # Performance comparison
            performance_diff = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                          'sensitivity', 'specificity']:
                quantum_val = quantum_metrics[metric]
                classical_val = classical_metrics[metric]
                performance_diff[metric] = quantum_val - classical_val
            
            comparison['performance_summary'][classical_name] = performance_diff
            
            # Efficiency comparison
            efficiency = {
                'training_time_ratio': quantum_results['training_time'] / classical_result['training_time'],
                'prediction_time_ratio': quantum_results['prediction_time'] / classical_result['prediction_time']
            }
            comparison['efficiency_analysis'][classical_name] = efficiency
        
        # Overall quantum advantages
        best_classical_accuracy = max([r['metrics']['accuracy'] for r in classical_results.values()])
        quantum_accuracy = quantum_metrics['accuracy']
        
        comparison['quantum_advantages'] = {
            'accuracy_improvement': quantum_accuracy - best_classical_accuracy,
            'beats_best_classical': quantum_accuracy > best_classical_accuracy,
            'medical_grade_performance': self._check_medical_grade(quantum_metrics),
            'quantum_advantage_score': self._calculate_quantum_advantage_score(
                quantum_results, classical_results
            )
        }
        
        return comparison
    
    def _test_statistical_significance(self, quantum_results: Dict, 
                                     classical_results: Dict) -> Dict:
        """Test statistical significance of quantum vs classical performance"""
        
        significance_tests = {}
        
        # Get quantum performance (single value)
        quantum_accuracy = quantum_results['metrics']['accuracy']
        
        # Compare against each classical model with cross-validation
        for classical_name, classical_result in classical_results.items():
            if classical_result['cv_scores'] is not None:
                cv_scores = np.array(classical_result['cv_scores'])
                
                # One-sample t-test (quantum vs classical CV mean)
                t_stat, p_value = stats.ttest_1samp(cv_scores, quantum_accuracy)
                
                # Effect size (Cohen's d)
                effect_size = (quantum_accuracy - np.mean(cv_scores)) / np.std(cv_scores)
                
                significance_tests[classical_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05,
                    'quantum_better': quantum_accuracy > np.mean(cv_scores),
                    'confidence_interval': stats.t.interval(
                        0.95, len(cv_scores)-1, 
                        loc=np.mean(cv_scores), 
                        scale=stats.sem(cv_scores)
                    )
                }
        
        return significance_tests
    
    def _validate_medical_requirements(self, quantum_results: Dict, 
                                     classical_results: Dict) -> Dict:
        """Validate models against medical AI requirements"""
        
        validation = {
            'quantum_validation': {},
            'classical_validation': {},
            'medical_grade_summary': {}
        }
        
        # Validate quantum model
        quantum_metrics = quantum_results['metrics']
        validation['quantum_validation'] = self._check_medical_requirements(quantum_metrics)
        
        # Validate classical models
        for name, result in classical_results.items():
            validation['classical_validation'][name] = self._check_medical_requirements(
                result['metrics']
            )
        
        # Summary
        quantum_passes = validation['quantum_validation']['passes_all_requirements']
        classical_passes = [v['passes_all_requirements'] for v in validation['classical_validation'].values()]
        
        validation['medical_grade_summary'] = {
            'quantum_medical_grade': quantum_passes,
            'classical_medical_grade_count': sum(classical_passes),
            'total_classical_models': len(classical_passes),
            'quantum_advantage_medical': quantum_passes and not any(classical_passes)
        }
        
        return validation
    
    def _check_medical_requirements(self, metrics: Dict) -> Dict:
        """Check if model meets medical AI requirements"""
        
        requirements = {
            'accuracy': metrics['accuracy'] >= self.medical_thresholds['minimum_accuracy'],
            'sensitivity': metrics['sensitivity'] >= self.medical_thresholds['minimum_sensitivity'],
            'specificity': metrics['specificity'] >= self.medical_thresholds['minimum_specificity'],
            'false_negative_rate': metrics['false_negative_rate'] <= self.medical_thresholds['maximum_false_negative']
        }
        
        passes_all = all(requirements.values())
        
        return {
            'individual_requirements': requirements,
            'passes_all_requirements': passes_all,
            'medical_grade_score': sum(requirements.values()) / len(requirements),
            'critical_failures': [k for k, v in requirements.items() if not v]
        }
    
    def _check_medical_grade(self, metrics: Dict) -> bool:
        """Quick check if model meets medical grade requirements"""
        return (metrics['accuracy'] >= self.medical_thresholds['minimum_accuracy'] and
                metrics['sensitivity'] >= self.medical_thresholds['minimum_sensitivity'] and
                metrics['specificity'] >= self.medical_thresholds['minimum_specificity'])
    
    def _calculate_quantum_advantage_score(self, quantum_results: Dict, 
                                         classical_results: Dict) -> float:
        """Calculate overall quantum advantage score"""
        
        quantum_metrics = quantum_results['metrics']
        
        # Performance score (vs best classical)
        best_classical_metrics = max(classical_results.values(), 
                                   key=lambda x: x['metrics']['accuracy'])['metrics']
        
        performance_score = 0.0
        weights = {'accuracy': 0.3, 'sensitivity': 0.3, 'specificity': 0.2, 'f1_score': 0.2}
        
        for metric, weight in weights.items():
            quantum_val = quantum_metrics[metric]
            classical_val = best_classical_metrics[metric]
            performance_score += weight * max(0, quantum_val - classical_val)
        
        # Efficiency penalty
        avg_classical_time = np.mean([r['training_time'] + r['prediction_time'] 
                                    for r in classical_results.values()])
        quantum_time = quantum_results['training_time'] + quantum_results['prediction_time']
        efficiency_penalty = max(0, (quantum_time / avg_classical_time - 1) * 0.1)
        
        # Medical grade bonus
        medical_bonus = 0.1 if self._check_medical_grade(quantum_metrics) else 0.0
        
        return max(0, performance_score - efficiency_penalty + medical_bonus)
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        
        if not self.benchmark_results:
            return "No benchmark results available. Run benchmark_quantum_vs_classical first."
        
        results = self.benchmark_results
        
        report = f"""
[DNA] Q-MediScan Quantum vs Classical Benchmark Report
═══════════════════════════════════════════════════

[CHART] PERFORMANCE SUMMARY
─────────────────────
Quantum Model Performance:
  • Accuracy: {results['quantum_results']['metrics']['accuracy']:.3f}
  • Sensitivity: {results['quantum_results']['metrics']['sensitivity']:.3f}
  • Specificity: {results['quantum_results']['metrics']['specificity']:.3f}
  • F1-Score: {results['quantum_results']['metrics']['f1_score']:.3f}
  • ROC AUC: {results['quantum_results']['metrics']['roc_auc']:.3f}

Classical Models Comparison:
"""
        
        for name, result in results['classical_results'].items():
            metrics = result['metrics']
            report += f"  {name}:\n"
            report += f"    • Accuracy: {metrics['accuracy']:.3f}\n"
            report += f"    • Sensitivity: {metrics['sensitivity']:.3f}\n"
            report += f"    • Specificity: {metrics['specificity']:.3f}\n"
        
        # Quantum advantage analysis
        qa_score = results['comparison']['quantum_advantages']['quantum_advantage_score']
        beats_classical = results['comparison']['quantum_advantages']['beats_best_classical']
        
        report += f"""
[ATOM]  QUANTUM ADVANTAGE ANALYSIS
─────────────────────────────
Quantum Advantage Score: {qa_score:.3f}
Beats Best Classical: {'[OK] YES' if beats_classical else '[ERROR] NO'}
Medical Grade Performance: {'[OK] YES' if results['comparison']['quantum_advantages']['medical_grade_performance'] else '[ERROR] NO'}

"""
        
        # Statistical significance
        report += "[GRAPH] STATISTICAL SIGNIFICANCE\n"
        report += "─────────────────────────\n"
        
        for name, test in results['statistical_significance'].items():
            significance = "[OK] Significant" if test['significant'] else "[ERROR] Not Significant"
            better = "[OK] Better" if test['quantum_better'] else "[ERROR] Worse"
            report += f"{name}: {significance}, {better} (p={test['p_value']:.4f})\n"
        
        # Medical validation
        medical_val = results['medical_validation']
        quantum_medical = medical_val['quantum_validation']['passes_all_requirements']
        classical_medical = medical_val['medical_grade_summary']['classical_medical_grade_count']
        
        report += f"""
[STETHOSCOPE] MEDICAL AI VALIDATION
──────────────────────
Quantum Model Medical Grade: {'[OK] PASS' if quantum_medical else '[ERROR] FAIL'}
Classical Models Medical Grade: {classical_medical}/{medical_val['medical_grade_summary']['total_classical_models']} models pass

Medical Requirements:
  • Minimum Accuracy (85%): {'[OK]' if results['quantum_results']['metrics']['accuracy'] >= 0.85 else '[ERROR]'}
  • Minimum Sensitivity (90%): {'[OK]' if results['quantum_results']['metrics']['sensitivity'] >= 0.90 else '[ERROR]'}
  • Minimum Specificity (80%): {'[OK]' if results['quantum_results']['metrics']['specificity'] >= 0.80 else '[ERROR]'}
  • Maximum False Negatives (10%): {'[OK]' if results['quantum_results']['metrics']['false_negative_rate'] <= 0.10 else '[ERROR]'}

⏱️  EFFICIENCY ANALYSIS
─────────────────────
Training Time: {results['quantum_results']['training_time']:.2f}s
Prediction Time: {results['quantum_results']['prediction_time']:.2f}s

[DART] RECOMMENDATIONS
─────────────────
"""
        
        # Generate recommendations
        if qa_score > 0.05 and quantum_medical:
            report += "[OK] STRONG RECOMMENDATION: Deploy quantum model for medical AI\n"
            report += "   Quantum model shows significant advantage with medical-grade performance.\n"
        elif qa_score > 0.02:
            report += "[WARNING]  MODERATE RECOMMENDATION: Consider quantum model for specialized cases\n"
            report += "   Quantum model shows promise but may need further optimization.\n"
        else:
            report += "[ERROR] NOT RECOMMENDED: Continue classical approach\n"
            report += "   Quantum model does not show sufficient advantage over classical methods.\n"
        
        report += f"""
[CLIPBOARD] TECHNICAL DETAILS
──────────────────
• Quantum Qubits: {results['quantum_results'].get('qubits', 'N/A')}
• Training Samples: {len(results['quantum_results']['predictions'])}
• Test Samples: {len(results['quantum_results']['predictions'])}
• Random State: {self.random_state}

════════════════════════════════════════════════════
Report generated by Q-MediScan Quantum Benchmarking System
"""
        
        return report
    
    def save_benchmark_results(self, filepath: str):
        """Save benchmark results to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.benchmark_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"[OK] Benchmark results saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

# Example usage and testing
if __name__ == "__main__":
    print("[TEST] Testing Quantum Medical Benchmarking System")
    print("=" * 50)
    
    # Generate synthetic medical data
    np.random.seed(42)
    n_samples = 200
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create mock quantum model for testing
    class MockQuantumModel:
        def __init__(self):
            self.is_trained = False
            
        def train(self, X, y):
            self.is_trained = True
            return {'status': 'trained', 'accuracy': 0.87}
            
        def predict(self, X):
            # Mock prediction with realistic medical AI performance
            prob = 0.75 + 0.2 * np.random.random()
            prediction = "Low Risk" if prob > 0.5 else "High Risk"
            return {
                'prediction': prediction,
                'probability': prob,
                'confidence': 0.85
            }
    
    # Initialize benchmark system
    benchmark = QuantumMedicalBenchmark(random_state=42)
    
    # Create mock quantum model
    quantum_model = MockQuantumModel()
    
    # Run comprehensive benchmark
    print("[MICROSCOPE] Running comprehensive benchmark...")
    results = benchmark.benchmark_quantum_vs_classical(
        quantum_model, X_train, X_test, y_train, y_test
    )
    
    # Generate and display report
    report = benchmark.generate_benchmark_report()
    print("\n" + report)
    
    # Save results
    benchmark.save_benchmark_results("benchmark_results.json")
    
    print("\n[OK] Quantum benchmarking system testing complete!")
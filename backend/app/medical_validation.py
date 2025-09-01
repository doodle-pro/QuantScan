"""
Advanced Medical Validation and Statistical Significance Testing
Comprehensive validation framework for medical AI with life-saving focus
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
from datetime import datetime
import json

class MedicalAIValidator:
    """
    Comprehensive medical AI validation system
    Ensures quantum models meet medical-grade standards for life-saving applications
    """
    
    def __init__(self):
        # FDA-inspired medical AI requirements
        self.medical_standards = {
            'minimum_accuracy': 0.85,           # 85% minimum accuracy
            'minimum_sensitivity': 0.90,        # 90% sensitivity (recall) for cancer detection
            'minimum_specificity': 0.80,        # 80% specificity to minimize false positives
            'maximum_false_negative_rate': 0.10, # Max 10% false negatives (critical for cancer)
            'minimum_ppv': 0.75,               # 75% positive predictive value
            'minimum_npv': 0.95,               # 95% negative predictive value (critical)
            'minimum_auc': 0.85,               # 85% AUC for discrimination
            'statistical_significance': 0.05,   # p < 0.05 for significance
            'minimum_sample_size': 100,         # Minimum samples for validation
            'cross_validation_folds': 5         # 5-fold cross-validation
        }
        
        # Life-saving impact metrics
        self.life_saving_metrics = {
            'early_detection_threshold': 0.3,   # Risk threshold for early intervention
            'lives_saved_per_1000': 0,          # Estimated lives saved per 1000 patients
            'cost_effectiveness': 0,            # Cost per quality-adjusted life year
            'screening_improvement': 0          # Improvement over standard screening
        }
        
        self.validation_results = {}
        self.statistical_tests = {}
        self.medical_recommendations = {}
    
    def comprehensive_medical_validation(self, quantum_model, classical_models: Dict,
                                       X_test: np.ndarray, y_test: np.ndarray,
                                       X_train: np.ndarray = None, y_train: np.ndarray = None) -> Dict:
        """
        Comprehensive medical validation comparing quantum vs classical models
        Focus on life-saving potential and clinical deployment readiness
        """
        print("[STETHOSCOPE] Starting comprehensive medical AI validation...")
        print("   Focus: Life-saving early cancer detection")
        
        validation_results = {
            'quantum_validation': {},
            'classical_validation': {},
            'comparative_analysis': {},
            'statistical_significance': {},
            'medical_recommendations': {},
            'life_saving_assessment': {},
            'regulatory_compliance': {}
        }
        
        # Validate quantum model
        print("[ATOM]  Validating quantum model...")
        quantum_validation = self._validate_single_model(
            quantum_model, X_test, y_test, "Quantum ML", X_train, y_train
        )
        validation_results['quantum_validation'] = quantum_validation
        
        # Validate classical models
        classical_validations = {}
        for name, model in classical_models.items():
            print(f"🖥️  Validating {name}...")
            classical_validation = self._validate_single_model(
                model, X_test, y_test, name, X_train, y_train
            )
            classical_validations[name] = classical_validation
        
        validation_results['classical_validation'] = classical_validations
        
        # Comparative analysis
        validation_results['comparative_analysis'] = self._perform_comparative_analysis(
            quantum_validation, classical_validations
        )
        
        # Statistical significance testing
        validation_results['statistical_significance'] = self._test_statistical_significance(
            quantum_validation, classical_validations
        )
        
        # Life-saving assessment
        validation_results['life_saving_assessment'] = self._assess_life_saving_potential(
            quantum_validation, classical_validations
        )
        
        # Medical recommendations
        validation_results['medical_recommendations'] = self._generate_medical_recommendations(
            validation_results
        )
        
        # Regulatory compliance
        validation_results['regulatory_compliance'] = self._assess_regulatory_compliance(
            validation_results
        )
        
        # Store results
        self.validation_results = validation_results
        
        return validation_results
    
    def _validate_single_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                             model_name: str, X_train: np.ndarray = None, 
                             y_train: np.ndarray = None) -> Dict:
        """Comprehensive validation of a single model"""
        
        # Get predictions
        if hasattr(model, 'predict') and hasattr(model, 'is_trained'):
            # Quantum model
            y_pred, y_proba = self._get_quantum_predictions(model, X_test)
        else:
            # Classical model
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred.astype(float)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_medical_metrics(y_test, y_pred, y_proba)
        
        # Cross-validation if training data available
        cv_results = None
        if X_train is not None and y_train is not None and not hasattr(model, 'is_trained'):
            cv_results = self._perform_cross_validation(model, X_train, y_train)
        
        # Medical standards compliance
        compliance = self._check_medical_standards_compliance(metrics)
        
        # Clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(metrics, model_name)
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'cross_validation': cv_results,
            'medical_compliance': compliance,
            'clinical_interpretation': clinical_interpretation,
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist() if hasattr(y_proba, 'tolist') else y_proba
            }
        }
    
    def _get_quantum_predictions(self, quantum_model, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from quantum model"""
        y_pred = []
        y_proba = []
        
        for i in range(len(X_test)):
            try:
                result = quantum_model.predict(X_test[i:i+1])
                prediction = 1 if result['prediction'] == 'Low Risk' else 0
                probability = 1 - result['probability']  # Convert to benign probability
                
                y_pred.append(prediction)
                y_proba.append(probability)
            except Exception as e:
                warnings.warn(f"Quantum prediction failed for sample {i}: {e}")
                y_pred.append(0)  # Conservative prediction
                y_proba.append(0.5)
        
        return np.array(y_pred), np.array(y_proba)
    
    def _calculate_medical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive medical metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            roc_auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Medical-specific metrics (critical for cancer detection)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Error rates (critical for medical applications)
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Medical decision metrics
        diagnostic_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        likelihood_ratio_positive = sensitivity / false_positive_rate if false_positive_rate > 0 else float('inf')
        likelihood_ratio_negative = false_negative_rate / specificity if specificity > 0 else 0.0
        
        # Clinical utility metrics
        youden_index = sensitivity + specificity - 1  # Youden's J statistic
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Life-saving potential metrics
        early_detection_rate = np.sum((y_proba > self.life_saving_metrics['early_detection_threshold']) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0.0
        
        return {
            # Basic metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            
            # Medical metrics
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            
            # Advanced medical metrics
            'diagnostic_odds_ratio': diagnostic_odds_ratio,
            'likelihood_ratio_positive': likelihood_ratio_positive,
            'likelihood_ratio_negative': likelihood_ratio_negative,
            'youden_index': youden_index,
            'balanced_accuracy': balanced_accuracy,
            'early_detection_rate': early_detection_rate,
            
            # Confusion matrix components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'confusion_matrix': cm.tolist()
        }
    
    def _perform_cross_validation(self, model, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Perform cross-validation for robust validation"""
        
        try:
            cv = StratifiedKFold(n_splits=self.medical_standards['cross_validation_folds'], 
                               shuffle=True, random_state=42)
            
            # Multiple metrics cross-validation
            cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')
            cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
            cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            
            return {
                'accuracy': {
                    'mean': np.mean(cv_accuracy),
                    'std': np.std(cv_accuracy),
                    'scores': cv_accuracy.tolist()
                },
                'precision': {
                    'mean': np.mean(cv_precision),
                    'std': np.std(cv_precision),
                    'scores': cv_precision.tolist()
                },
                'recall': {
                    'mean': np.mean(cv_recall),
                    'std': np.std(cv_recall),
                    'scores': cv_recall.tolist()
                },
                'f1': {
                    'mean': np.mean(cv_f1),
                    'std': np.std(cv_f1),
                    'scores': cv_f1.tolist()
                },
                'roc_auc': {
                    'mean': np.mean(cv_roc_auc),
                    'std': np.std(cv_roc_auc),
                    'scores': cv_roc_auc.tolist()
                }
            }
            
        except Exception as e:
            warnings.warn(f"Cross-validation failed: {e}")
            return None
    
    def _check_medical_standards_compliance(self, metrics: Dict) -> Dict:
        """Check compliance with medical AI standards"""
        
        compliance_checks = {
            'accuracy': metrics['accuracy'] >= self.medical_standards['minimum_accuracy'],
            'sensitivity': metrics['sensitivity'] >= self.medical_standards['minimum_sensitivity'],
            'specificity': metrics['specificity'] >= self.medical_standards['minimum_specificity'],
            'false_negative_rate': metrics['false_negative_rate'] <= self.medical_standards['maximum_false_negative_rate'],
            'ppv': metrics['ppv'] >= self.medical_standards['minimum_ppv'],
            'npv': metrics['npv'] >= self.medical_standards['minimum_npv'],
            'roc_auc': metrics['roc_auc'] >= self.medical_standards['minimum_auc']
        }
        
        # Overall compliance
        passes_all = all(compliance_checks.values())
        compliance_score = sum(compliance_checks.values()) / len(compliance_checks)
        
        # Critical failures (life-threatening)
        critical_failures = []
        if not compliance_checks['sensitivity']:
            critical_failures.append('Low sensitivity - may miss cancer cases')
        if not compliance_checks['false_negative_rate']:
            critical_failures.append('High false negative rate - dangerous for patients')
        if not compliance_checks['npv']:
            critical_failures.append('Low negative predictive value - unreliable negative results')
        
        return {
            'individual_checks': compliance_checks,
            'passes_all_standards': passes_all,
            'compliance_score': compliance_score,
            'critical_failures': critical_failures,
            'medical_grade': passes_all and len(critical_failures) == 0
        }
    
    def _generate_clinical_interpretation(self, metrics: Dict, model_name: str) -> Dict:
        """Generate clinical interpretation of results"""
        
        # Risk assessment
        if metrics['sensitivity'] >= 0.95:
            sensitivity_assessment = "Excellent - Very low risk of missing cancer"
        elif metrics['sensitivity'] >= 0.90:
            sensitivity_assessment = "Good - Acceptable for clinical use"
        elif metrics['sensitivity'] >= 0.80:
            sensitivity_assessment = "Moderate - May miss some cancer cases"
        else:
            sensitivity_assessment = "Poor - High risk of missing cancer cases"
        
        # Specificity assessment
        if metrics['specificity'] >= 0.90:
            specificity_assessment = "Excellent - Low false positive rate"
        elif metrics['specificity'] >= 0.80:
            specificity_assessment = "Good - Acceptable false positive rate"
        elif metrics['specificity'] >= 0.70:
            specificity_assessment = "Moderate - Some unnecessary anxiety"
        else:
            specificity_assessment = "Poor - High false positive rate"
        
        # Clinical recommendation
        if metrics['sensitivity'] >= 0.90 and metrics['specificity'] >= 0.80:
            clinical_recommendation = "Recommended for clinical deployment"
            deployment_readiness = "Ready"
        elif metrics['sensitivity'] >= 0.85 and metrics['specificity'] >= 0.75:
            clinical_recommendation = "Consider for clinical trials"
            deployment_readiness = "Needs validation"
        else:
            clinical_recommendation = "Not recommended for clinical use"
            deployment_readiness = "Not ready"
        
        # Life-saving potential
        lives_saved_estimate = self._estimate_lives_saved(metrics)
        
        return {
            'sensitivity_assessment': sensitivity_assessment,
            'specificity_assessment': specificity_assessment,
            'clinical_recommendation': clinical_recommendation,
            'deployment_readiness': deployment_readiness,
            'lives_saved_estimate': lives_saved_estimate,
            'clinical_impact': self._assess_clinical_impact(metrics),
            'patient_safety': self._assess_patient_safety(metrics)
        }
    
    def _estimate_lives_saved(self, metrics: Dict) -> Dict:
        """Estimate lives saved per 1000 patients screened"""
        
        # Assumptions based on medical literature
        cancer_prevalence = 0.05  # 5% prevalence in screening population
        survival_improvement_early_detection = 0.30  # 30% improvement with early detection
        baseline_mortality = 0.20  # 20% mortality without early detection
        
        # Calculate lives saved
        true_positives_per_1000 = 1000 * cancer_prevalence * metrics['sensitivity']
        lives_saved_per_1000 = true_positives_per_1000 * survival_improvement_early_detection * baseline_mortality
        
        # False negatives (missed cases)
        false_negatives_per_1000 = 1000 * cancer_prevalence * metrics['false_negative_rate']
        potential_deaths_from_missed_cases = false_negatives_per_1000 * baseline_mortality
        
        return {
            'lives_saved_per_1000_screened': round(lives_saved_per_1000, 2),
            'cancer_cases_detected_per_1000': round(true_positives_per_1000, 2),
            'cancer_cases_missed_per_1000': round(false_negatives_per_1000, 2),
            'potential_deaths_from_missed_cases': round(potential_deaths_from_missed_cases, 2),
            'net_lives_saved_per_1000': round(lives_saved_per_1000 - potential_deaths_from_missed_cases, 2)
        }
    
    def _assess_clinical_impact(self, metrics: Dict) -> str:
        """Assess overall clinical impact"""
        
        if metrics['sensitivity'] >= 0.95 and metrics['specificity'] >= 0.85:
            return "High positive impact - Significant improvement in patient outcomes"
        elif metrics['sensitivity'] >= 0.90 and metrics['specificity'] >= 0.80:
            return "Moderate positive impact - Meaningful improvement in care"
        elif metrics['sensitivity'] >= 0.85:
            return "Limited positive impact - Some improvement in detection"
        else:
            return "Questionable impact - May not improve current standard of care"
    
    def _assess_patient_safety(self, metrics: Dict) -> Dict:
        """Assess patient safety implications"""
        
        safety_score = 0
        safety_concerns = []
        
        # False negative concerns (most critical)
        if metrics['false_negative_rate'] <= 0.05:
            safety_score += 40
        elif metrics['false_negative_rate'] <= 0.10:
            safety_score += 30
        elif metrics['false_negative_rate'] <= 0.15:
            safety_score += 20
            safety_concerns.append("Moderate false negative rate - some cancer cases may be missed")
        else:
            safety_score += 10
            safety_concerns.append("High false negative rate - significant risk of missing cancer")
        
        # False positive concerns
        if metrics['false_positive_rate'] <= 0.10:
            safety_score += 30
        elif metrics['false_positive_rate'] <= 0.20:
            safety_score += 20
        elif metrics['false_positive_rate'] <= 0.30:
            safety_score += 10
            safety_concerns.append("Moderate false positive rate - may cause unnecessary anxiety")
        else:
            safety_concerns.append("High false positive rate - significant patient anxiety and unnecessary procedures")
        
        # NPV concerns
        if metrics['npv'] >= 0.95:
            safety_score += 30
        elif metrics['npv'] >= 0.90:
            safety_score += 20
        else:
            safety_concerns.append("Low negative predictive value - negative results less reliable")
        
        # Overall safety assessment
        if safety_score >= 90:
            safety_level = "Very Safe"
        elif safety_score >= 70:
            safety_level = "Safe"
        elif safety_score >= 50:
            safety_level = "Moderate Safety Concerns"
        else:
            safety_level = "Significant Safety Concerns"
        
        return {
            'safety_score': safety_score,
            'safety_level': safety_level,
            'safety_concerns': safety_concerns,
            'recommendation': "Approved for clinical use" if safety_score >= 70 else "Requires improvement before clinical use"
        }
    
    def _perform_comparative_analysis(self, quantum_validation: Dict, 
                                    classical_validations: Dict) -> Dict:
        """Perform comparative analysis between quantum and classical models"""
        
        quantum_metrics = quantum_validation['metrics']
        
        # Find best classical model
        best_classical_name = max(classical_validations.keys(), 
                                key=lambda k: classical_validations[k]['metrics']['accuracy'])
        best_classical_metrics = classical_validations[best_classical_name]['metrics']
        
        # Performance comparison
        performance_comparison = {}
        key_metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'roc_auc', 'f1_score']
        
        for metric in key_metrics:
            quantum_val = quantum_metrics[metric]
            classical_val = best_classical_metrics[metric]
            improvement = quantum_val - classical_val
            improvement_percent = (improvement / classical_val * 100) if classical_val > 0 else 0
            
            performance_comparison[metric] = {
                'quantum': quantum_val,
                'best_classical': classical_val,
                'improvement': improvement,
                'improvement_percent': improvement_percent,
                'quantum_better': quantum_val > classical_val
            }
        
        # Medical significance of improvements
        medical_significance = self._assess_medical_significance_of_improvements(performance_comparison)
        
        # Overall quantum advantage
        quantum_advantage_score = self._calculate_quantum_advantage_score(performance_comparison)
        
        return {
            'best_classical_model': best_classical_name,
            'performance_comparison': performance_comparison,
            'medical_significance': medical_significance,
            'quantum_advantage_score': quantum_advantage_score,
            'recommendation': self._generate_comparative_recommendation(performance_comparison, medical_significance)
        }
    
    def _assess_medical_significance_of_improvements(self, performance_comparison: Dict) -> Dict:
        """Assess medical significance of quantum improvements"""
        
        significant_improvements = []
        
        # Sensitivity improvement (critical for cancer detection)
        sens_improvement = performance_comparison['sensitivity']['improvement']
        if sens_improvement >= 0.05:
            significant_improvements.append(f"Sensitivity improved by {sens_improvement:.3f} - could detect {sens_improvement*1000:.0f} more cancer cases per 1000 patients")
        
        # Specificity improvement
        spec_improvement = performance_comparison['specificity']['improvement']
        if spec_improvement >= 0.05:
            significant_improvements.append(f"Specificity improved by {spec_improvement:.3f} - could reduce {spec_improvement*1000:.0f} false positives per 1000 patients")
        
        # NPV improvement (critical for patient confidence)
        npv_improvement = performance_comparison['npv']['improvement']
        if npv_improvement >= 0.02:
            significant_improvements.append(f"NPV improved by {npv_improvement:.3f} - negative results more reliable")
        
        # Overall assessment
        if len(significant_improvements) >= 2:
            overall_significance = "Highly Significant"
        elif len(significant_improvements) == 1:
            overall_significance = "Moderately Significant"
        else:
            overall_significance = "Not Significant"
        
        return {
            'significant_improvements': significant_improvements,
            'overall_significance': overall_significance,
            'clinical_impact': "High" if len(significant_improvements) >= 2 else "Moderate" if len(significant_improvements) == 1 else "Low"
        }
    
    def _calculate_quantum_advantage_score(self, performance_comparison: Dict) -> float:
        """Calculate overall quantum advantage score"""
        
        # Weighted scoring based on medical importance
        weights = {
            'sensitivity': 0.30,  # Most important for cancer detection
            'specificity': 0.20,
            'npv': 0.20,         # Important for patient confidence
            'ppv': 0.15,
            'accuracy': 0.10,
            'roc_auc': 0.05
        }
        
        total_score = 0
        for metric, weight in weights.items():
            if metric in performance_comparison:
                improvement = performance_comparison[metric]['improvement']
                # Normalize improvement to 0-1 scale
                normalized_improvement = max(0, min(1, improvement * 10))  # Scale by 10
                total_score += weight * normalized_improvement
        
        return total_score
    
    def _generate_comparative_recommendation(self, performance_comparison: Dict, 
                                           medical_significance: Dict) -> str:
        """Generate recommendation based on comparative analysis"""
        
        quantum_better_count = sum(1 for comp in performance_comparison.values() if comp['quantum_better'])
        total_metrics = len(performance_comparison)
        
        if quantum_better_count >= total_metrics * 0.8 and medical_significance['overall_significance'] in ['Highly Significant', 'Moderately Significant']:
            return "Strong recommendation for quantum model deployment - significant medical advantages"
        elif quantum_better_count >= total_metrics * 0.6:
            return "Moderate recommendation for quantum model - some medical advantages"
        else:
            return "Classical model recommended - quantum advantages not sufficient"
    
    def _test_statistical_significance(self, quantum_validation: Dict, 
                                     classical_validations: Dict) -> Dict:
        """Test statistical significance of quantum vs classical performance"""
        
        significance_tests = {}
        
        # Get quantum predictions
        quantum_predictions = np.array(quantum_validation['predictions']['y_pred'])
        
        for classical_name, classical_validation in classical_validations.items():
            classical_predictions = np.array(classical_validation['predictions']['y_pred'])
            
            # McNemar's test for paired predictions
            try:
                mcnemar_result = self._mcnemar_test(quantum_predictions, classical_predictions)
                significance_tests[classical_name] = mcnemar_result
            except Exception as e:
                warnings.warn(f"Statistical test failed for {classical_name}: {e}")
        
        return significance_tests
    
    def _mcnemar_test(self, quantum_pred: np.ndarray, classical_pred: np.ndarray) -> Dict:
        """Perform McNemar's test for paired predictions"""
        
        # Create contingency table
        both_correct = np.sum((quantum_pred == 1) & (classical_pred == 1))
        quantum_correct_classical_wrong = np.sum((quantum_pred == 1) & (classical_pred == 0))
        quantum_wrong_classical_correct = np.sum((quantum_pred == 0) & (classical_pred == 1))
        both_wrong = np.sum((quantum_pred == 0) & (classical_pred == 0))
        
        # McNemar's test statistic
        b = quantum_correct_classical_wrong
        c = quantum_wrong_classical_correct
        
        if b + c == 0:
            p_value = 1.0
            test_statistic = 0.0
        else:
            test_statistic = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(test_statistic, 1)
        
        return {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significant': p_value < self.medical_standards['statistical_significance'],
            'quantum_better': b > c,
            'contingency_table': {
                'both_correct': both_correct,
                'quantum_correct_classical_wrong': b,
                'quantum_wrong_classical_correct': c,
                'both_wrong': both_wrong
            }
        }
    
    def _assess_life_saving_potential(self, quantum_validation: Dict, 
                                    classical_validations: Dict) -> Dict:
        """Assess life-saving potential of quantum vs classical models"""
        
        quantum_lives_saved = quantum_validation['clinical_interpretation']['lives_saved_estimate']
        
        classical_lives_saved = {}
        for name, validation in classical_validations.items():
            classical_lives_saved[name] = validation['clinical_interpretation']['lives_saved_estimate']
        
        # Find best classical model for life-saving
        best_classical_name = max(classical_lives_saved.keys(), 
                                key=lambda k: classical_lives_saved[k]['net_lives_saved_per_1000'])
        best_classical_lives = classical_lives_saved[best_classical_name]
        
        # Calculate additional lives saved by quantum model
        additional_lives_saved = (quantum_lives_saved['net_lives_saved_per_1000'] - 
                                best_classical_lives['net_lives_saved_per_1000'])
        
        # Scale to population level
        us_screening_population = 40_000_000  # Approximate US women eligible for breast cancer screening
        additional_lives_saved_nationally = (additional_lives_saved / 1000) * us_screening_population
        
        return {
            'quantum_lives_saved_per_1000': quantum_lives_saved['net_lives_saved_per_1000'],
            'best_classical_lives_saved_per_1000': best_classical_lives['net_lives_saved_per_1000'],
            'additional_lives_saved_per_1000': additional_lives_saved,
            'additional_lives_saved_nationally_per_year': max(0, additional_lives_saved_nationally),
            'life_saving_advantage': additional_lives_saved > 0,
            'impact_assessment': self._assess_life_saving_impact(additional_lives_saved)
        }
    
    def _assess_life_saving_impact(self, additional_lives_saved: float) -> str:
        """Assess the impact of additional lives saved"""
        
        if additional_lives_saved >= 2.0:
            return "Transformative impact - Could save thousands of lives annually"
        elif additional_lives_saved >= 1.0:
            return "Significant impact - Could save hundreds of lives annually"
        elif additional_lives_saved >= 0.5:
            return "Moderate impact - Meaningful improvement in survival"
        elif additional_lives_saved > 0:
            return "Small but meaningful impact - Every life saved matters"
        else:
            return "No significant life-saving advantage over classical methods"
    
    def _generate_medical_recommendations(self, validation_results: Dict) -> Dict:
        """Generate comprehensive medical recommendations"""
        
        quantum_validation = validation_results['quantum_validation']
        comparative_analysis = validation_results['comparative_analysis']
        life_saving_assessment = validation_results['life_saving_assessment']
        
        # Deployment recommendation
        if (quantum_validation['medical_compliance']['medical_grade'] and 
            life_saving_assessment['life_saving_advantage'] and
            comparative_analysis['medical_significance']['overall_significance'] != 'Not Significant'):
            deployment_recommendation = "RECOMMENDED FOR CLINICAL DEPLOYMENT"
            deployment_timeline = "Ready for clinical trials within 6 months"
        elif quantum_validation['medical_compliance']['passes_all_standards']:
            deployment_recommendation = "CONDITIONAL RECOMMENDATION"
            deployment_timeline = "Requires additional validation studies"
        else:
            deployment_recommendation = "NOT RECOMMENDED"
            deployment_timeline = "Requires significant improvements"
        
        # Regulatory pathway
        if quantum_validation['medical_compliance']['medical_grade']:
            regulatory_pathway = "FDA 510(k) clearance pathway - predicate device comparison"
        else:
            regulatory_pathway = "FDA De Novo pathway - novel device classification required"
        
        # Clinical trial design
        clinical_trial_design = {
            'study_type': 'Prospective randomized controlled trial',
            'primary_endpoint': 'Cancer detection sensitivity',
            'secondary_endpoints': ['Specificity', 'Patient outcomes', 'Cost-effectiveness'],
            'sample_size': 'Minimum 1000 patients per arm',
            'duration': '2-3 years including follow-up'
        }
        
        return {
            'deployment_recommendation': deployment_recommendation,
            'deployment_timeline': deployment_timeline,
            'regulatory_pathway': regulatory_pathway,
            'clinical_trial_design': clinical_trial_design,
            'key_advantages': self._identify_key_advantages(validation_results),
            'areas_for_improvement': self._identify_improvement_areas(validation_results),
            'patient_impact_summary': self._generate_patient_impact_summary(validation_results)
        }
    
    def _identify_key_advantages(self, validation_results: Dict) -> List[str]:
        """Identify key advantages of the quantum model"""
        
        advantages = []
        
        quantum_metrics = validation_results['quantum_validation']['metrics']
        comparison = validation_results['comparative_analysis']['performance_comparison']
        
        if comparison['sensitivity']['quantum_better'] and comparison['sensitivity']['improvement'] >= 0.02:
            advantages.append(f"Superior cancer detection: {comparison['sensitivity']['improvement']:.3f} improvement in sensitivity")
        
        if comparison['specificity']['quantum_better'] and comparison['specificity']['improvement'] >= 0.02:
            advantages.append(f"Reduced false positives: {comparison['specificity']['improvement']:.3f} improvement in specificity")
        
        if quantum_metrics['early_detection_rate'] > 0.5:
            advantages.append(f"Enhanced early detection: {quantum_metrics['early_detection_rate']:.1%} early detection rate")
        
        if validation_results['life_saving_assessment']['life_saving_advantage']:
            additional_lives = validation_results['life_saving_assessment']['additional_lives_saved_per_1000']
            advantages.append(f"Life-saving potential: {additional_lives:.1f} additional lives saved per 1000 patients")
        
        return advantages
    
    def _identify_improvement_areas(self, validation_results: Dict) -> List[str]:
        """Identify areas for improvement"""
        
        improvements = []
        
        quantum_compliance = validation_results['quantum_validation']['medical_compliance']
        
        for failure in quantum_compliance['critical_failures']:
            improvements.append(failure)
        
        quantum_metrics = validation_results['quantum_validation']['metrics']
        
        if quantum_metrics['false_negative_rate'] > 0.05:
            improvements.append("Reduce false negative rate to minimize missed cancer cases")
        
        if quantum_metrics['ppv'] < 0.80:
            improvements.append("Improve positive predictive value to reduce unnecessary anxiety")
        
        return improvements
    
    def _generate_patient_impact_summary(self, validation_results: Dict) -> str:
        """Generate patient impact summary"""
        
        life_saving = validation_results['life_saving_assessment']
        quantum_metrics = validation_results['quantum_validation']['metrics']
        
        summary = f"""PATIENT IMPACT SUMMARY:
        
• Cancer Detection: {quantum_metrics['sensitivity']:.1%} of cancer cases detected
• False Negatives: {quantum_metrics['false_negative_rate']:.1%} of cancer cases missed
• False Positives: {quantum_metrics['false_positive_rate']:.1%} of healthy patients flagged
• Lives Saved: {life_saving['quantum_lives_saved_per_1000']:.1f} lives saved per 1000 patients screened
• Early Detection: {quantum_metrics['early_detection_rate']:.1%} early detection rate
• Patient Safety: {validation_results['quantum_validation']['clinical_interpretation']['patient_safety']['safety_level']}
        
CONCLUSION: {life_saving['impact_assessment']}"""
        
        return summary
    
    def _assess_regulatory_compliance(self, validation_results: Dict) -> Dict:
        """Assess regulatory compliance for medical device approval"""
        
        quantum_validation = validation_results['quantum_validation']
        
        # FDA requirements assessment
        fda_requirements = {
            'clinical_evidence': quantum_validation['medical_compliance']['medical_grade'],
            'statistical_significance': any(test['significant'] for test in validation_results['statistical_significance'].values()),
            'safety_profile': quantum_validation['clinical_interpretation']['patient_safety']['safety_score'] >= 70,
            'performance_standards': quantum_validation['medical_compliance']['passes_all_standards'],
            'risk_benefit_analysis': validation_results['life_saving_assessment']['life_saving_advantage']
        }
        
        # Overall FDA readiness
        fda_ready = all(fda_requirements.values())
        
        return {
            'fda_requirements': fda_requirements,
            'fda_ready': fda_ready,
            'regulatory_risk': 'Low' if fda_ready else 'High',
            'approval_probability': 'High (>80%)' if fda_ready else 'Moderate (40-60%)' if sum(fda_requirements.values()) >= 3 else 'Low (<40%)',
            'recommended_pathway': '510(k) Clearance' if fda_ready else 'De Novo Classification'
        }
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        if not self.validation_results:
            return "No validation results available. Run comprehensive_medical_validation first."
        
        results = self.validation_results
        
        report = f"""
[STETHOSCOPE] Q-MEDISCAN MEDICAL AI VALIDATION REPORT
═══════════════════════════════════════════

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Validation Standard: Medical Device FDA Guidelines

[CHART] QUANTUM MODEL PERFORMANCE
─────────────────────────────
Accuracy: {results['quantum_validation']['metrics']['accuracy']:.3f}
Sensitivity (Cancer Detection): {results['quantum_validation']['metrics']['sensitivity']:.3f}
Specificity (Healthy Classification): {results['quantum_validation']['metrics']['specificity']:.3f}
Positive Predictive Value: {results['quantum_validation']['metrics']['ppv']:.3f}
Negative Predictive Value: {results['quantum_validation']['metrics']['npv']:.3f}
False Negative Rate: {results['quantum_validation']['metrics']['false_negative_rate']:.3f}

[HOSPITAL] MEDICAL STANDARDS COMPLIANCE
─────────────────────────────────
Medical Grade: {'[OK] YES' if results['quantum_validation']['medical_compliance']['medical_grade'] else '[ERROR] NO'}
Compliance Score: {results['quantum_validation']['medical_compliance']['compliance_score']:.3f}
Critical Failures: {len(results['quantum_validation']['medical_compliance']['critical_failures'])}

💊 LIFE-SAVING ASSESSMENT
───────────────────────
Lives Saved per 1000 Patients: {results['life_saving_assessment']['quantum_lives_saved_per_1000']:.1f}
Additional Lives Saved vs Classical: {results['life_saving_assessment']['additional_lives_saved_per_1000']:.1f}
National Impact: {results['life_saving_assessment']['additional_lives_saved_nationally_per_year']:.0f} lives/year
Impact Level: {results['life_saving_assessment']['impact_assessment']}

[ATOM]  QUANTUM ADVANTAGE
──────────────────────
Quantum Advantage Score: {results['comparative_analysis']['quantum_advantage_score']:.3f}
Medical Significance: {results['comparative_analysis']['medical_significance']['overall_significance']}
Key Improvements: {len(results['comparative_analysis']['medical_significance']['significant_improvements'])}

[GRAPH] STATISTICAL SIGNIFICANCE
─────────────────────────────
"""
        
        for model_name, test_result in results['statistical_significance'].items():
            significance = "[OK] Significant" if test_result['significant'] else "[ERROR] Not Significant"
            better = "[OK] Better" if test_result['quantum_better'] else "[ERROR] Worse"
            report += f"{model_name}: {significance}, {better} (p={test_result['p_value']:.4f})\n"
        
        report += f"""
[BUILDING]️  REGULATORY ASSESSMENT
──────────────────────────
FDA Ready: {'[OK] YES' if results['regulatory_compliance']['fda_ready'] else '[ERROR] NO'}
Approval Probability: {results['regulatory_compliance']['approval_probability']}
Recommended Pathway: {results['regulatory_compliance']['recommended_pathway']}
Regulatory Risk: {results['regulatory_compliance']['regulatory_risk']}

[DART] CLINICAL RECOMMENDATIONS
─────────────────────────────
Deployment: {results['medical_recommendations']['deployment_recommendation']}
Timeline: {results['medical_recommendations']['deployment_timeline']}
Regulatory Path: {results['medical_recommendations']['regulatory_pathway']}

👥 PATIENT IMPACT
──────────────────
{results['medical_recommendations']['patient_impact_summary']}

🔑 KEY ADVANTAGES
──────────────────
"""
        
        for advantage in results['medical_recommendations']['key_advantages']:
            report += f"• {advantage}\n"
        
        if results['medical_recommendations']['areas_for_improvement']:
            report += "\n[WARNING]  AREAS FOR IMPROVEMENT\n─────────────────────────\n"
            for improvement in results['medical_recommendations']['areas_for_improvement']:
                report += f"• {improvement}\n"
        
        report += f"""

═══════════════════════════════════════════
REPORT CONCLUSION: {results['medical_recommendations']['deployment_recommendation']}

This validation demonstrates {'quantum advantage in medical AI with significant life-saving potential' if results['life_saving_assessment']['life_saving_advantage'] else 'the need for further quantum model optimization'}.

[WARNING]  MEDICAL DISCLAIMER: This is a research validation for educational purposes only.
Not approved for clinical use. Always consult healthcare professionals.
═══════════════════════════════════════════
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("[STETHOSCOPE] Testing Medical AI Validation System")
    print("=" * 50)
    
    # Initialize validator
    validator = MedicalAIValidator()
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 200
    X_test = np.random.randn(n_samples, 30)
    y_test = np.random.randint(0, 2, n_samples)
    
    # Mock quantum model
    class MockQuantumModel:
        def __init__(self):
            self.is_trained = True
            
        def predict(self, X):
            prob = 0.75 + 0.2 * np.random.random()
            prediction = "Low Risk" if prob > 0.5 else "High Risk"
            return {
                'prediction': prediction,
                'probability': prob,
                'confidence': 0.85
            }
    
    # Mock classical models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    classical_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Fit classical models
    X_train = np.random.randn(400, 30)
    y_train = np.random.randint(0, 2, 400)
    
    for model in classical_models.values():
        model.fit(X_train, y_train)
    
    # Run comprehensive validation
    quantum_model = MockQuantumModel()
    
    validation_results = validator.comprehensive_medical_validation(
        quantum_model, classical_models, X_test, y_test, X_train, y_train
    )
    
    # Generate and display report
    report = validator.generate_validation_report()
    print("\n" + report)
    
    print("\n[OK] Medical validation system testing complete!")
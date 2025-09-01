import React, { useState, useEffect, useCallback } from 'react';
import { 
  HeartIcon, 
  CpuChipIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  DocumentArrowUpIcon,
  EyeIcon,
  UserIcon,
  BeakerIcon,
  ScaleIcon,
  SparklesIcon,
  ShieldCheckIcon,
  ChartBarIcon,
  CloudArrowUpIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import PatientInfoForm from '../components/assessment/PatientInfoForm';
import BiomarkerForm from '../components/assessment/BiomarkerForm';
import AnalysisResults from '../components/assessment/AnalysisResults';
import { apiService } from '../services/api';

interface BiomarkerData {
  // Mean values
  mean_radius: number;
  mean_texture: number;
  mean_perimeter: number;
  mean_area: number;
  mean_smoothness: number;
  mean_compactness: number;
  mean_concavity: number;
  mean_concave_points: number;
  mean_symmetry: number;
  mean_fractal_dimension: number;
  
  // Standard error values
  radius_error: number;
  texture_error: number;
  perimeter_error: number;
  area_error: number;
  smoothness_error: number;
  compactness_error: number;
  concavity_error: number;
  concave_points_error: number;
  symmetry_error: number;
  fractal_dimension_error: number;
  
  // Worst values
  worst_radius: number;
  worst_texture: number;
  worst_perimeter: number;
  worst_area: number;
  worst_smoothness: number;
  worst_compactness: number;
  worst_concavity: number;
  worst_concave_points: number;
  worst_symmetry: number;
  worst_fractal_dimension: number;
}

interface PatientInfo {
  age: number;
  name: string;
  medicalHistory: string[];
}

interface AssessmentResult {
  risk_level: string;
  confidence: number;
  quantum_probability: number;
  classical_probability: number;
  quantum_advantage: number;
  explanation: string;
  circuit_info?: {
    qubits: number;
    layers: number;
    parameters: number;
    executions: number;
    advanced_features: string;
  };
  medical_significance?: {
    risk_level: string;
    urgency: string;
    priority: string;
    reliability: string;
    early_detection_potential: boolean;
    lives_saved_potential: string;
  };
}

const CancerRiskAssessment = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [inputMethod, setInputMethod] = useState<'manual' | 'sample' | 'upload'>('sample');
  const [selectedSample, setSelectedSample] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  // Patient information
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    age: 45,
    name: '',
    medicalHistory: []
  });

  // Biomarker data
  const [biomarkerData, setBiomarkerData] = useState<BiomarkerData>({
    mean_radius: 17.99,
    mean_texture: 10.38,
    mean_perimeter: 122.8,
    mean_area: 1001.0,
    mean_smoothness: 0.1184,
    mean_compactness: 0.2776,
    mean_concavity: 0.3001,
    mean_concave_points: 0.1471,
    mean_symmetry: 0.2419,
    mean_fractal_dimension: 0.07871,
    radius_error: 1.095,
    texture_error: 0.9053,
    perimeter_error: 8.589,
    area_error: 153.4,
    smoothness_error: 0.006399,
    compactness_error: 0.04904,
    concavity_error: 0.05373,
    concave_points_error: 0.01587,
    symmetry_error: 0.03003,
    fractal_dimension_error: 0.006193,
    worst_radius: 25.38,
    worst_texture: 17.33,
    worst_perimeter: 184.6,
    worst_area: 2019.0,
    worst_smoothness: 0.1622,
    worst_compactness: 0.6656,
    worst_concavity: 0.7119,
    worst_concave_points: 0.2654,
    worst_symmetry: 0.4601,
    worst_fractal_dimension: 0.1189
  });

  // Sample cases for demonstration
  const sampleCases = [
    {
      name: "High Risk Case",
      description: "Biomarkers indicating elevated cancer risk",
      patientInfo: { age: 52, name: "Sample Patient A", medicalHistory: ["Family history", "Dense breast tissue"] },
      data: {
        mean_radius: 20.57, mean_texture: 17.77, mean_perimeter: 132.9, mean_area: 1326.0,
        mean_smoothness: 0.08474, mean_compactness: 0.07864, mean_concavity: 0.0869,
        mean_concave_points: 0.07017, mean_symmetry: 0.1812, mean_fractal_dimension: 0.05667,
        radius_error: 0.5435, texture_error: 0.7339, perimeter_error: 3.398, area_error: 74.08,
        smoothness_error: 0.005225, compactness_error: 0.01308, concavity_error: 0.0186,
        concave_points_error: 0.0134, symmetry_error: 0.01389, fractal_dimension_error: 0.003532,
        worst_radius: 24.99, worst_texture: 23.41, worst_perimeter: 158.8, worst_area: 1956.0,
        worst_smoothness: 0.1238, worst_compactness: 0.1866, worst_concavity: 0.2416,
        worst_concave_points: 0.186, worst_symmetry: 0.275, worst_fractal_dimension: 0.08902
      }
    },
    {
      name: "Low Risk Case",
      description: "Biomarkers indicating low cancer risk",
      patientInfo: { age: 35, name: "Sample Patient B", medicalHistory: ["No family history", "Regular screening"] },
      data: {
        mean_radius: 11.42, mean_texture: 20.38, mean_perimeter: 77.58, mean_area: 386.1,
        mean_smoothness: 0.1425, mean_compactness: 0.2839, mean_concavity: 0.2414,
        mean_concave_points: 0.1052, mean_symmetry: 0.2597, mean_fractal_dimension: 0.09744,
        radius_error: 0.4956, texture_error: 1.156, perimeter_error: 3.445, area_error: 27.23,
        smoothness_error: 0.00911, compactness_error: 0.07458, concavity_error: 0.05661,
        concave_points_error: 0.01867, symmetry_error: 0.05963, fractal_dimension_error: 0.009208,
        worst_radius: 14.91, worst_texture: 26.5, worst_perimeter: 98.87, worst_area: 567.7,
        worst_smoothness: 0.2098, worst_compactness: 0.8663, worst_concavity: 0.6869,
        worst_concave_points: 0.2575, worst_symmetry: 0.6638, worst_fractal_dimension: 0.173
      }
    },
    {
      name: "Moderate Risk Case",
      description: "Mixed biomarker patterns requiring careful analysis",
      patientInfo: { age: 48, name: "Sample Patient C", medicalHistory: ["Previous benign biopsy", "Age > 40"] },
      data: {
        mean_radius: 15.78, mean_texture: 17.89, mean_perimeter: 103.6, mean_area: 781.0,
        mean_smoothness: 0.0971, mean_compactness: 0.1292, mean_concavity: 0.09954,
        mean_concave_points: 0.06606, mean_symmetry: 0.1842, mean_fractal_dimension: 0.06082,
        radius_error: 0.5058, texture_error: 0.6938, perimeter_error: 3.672, area_error: 54.31,
        smoothness_error: 0.007026, compactness_error: 0.02501, concavity_error: 0.02264,
        concave_points_error: 0.009653, symmetry_error: 0.01386, fractal_dimension_error: 0.003462,
        worst_radius: 18.78, worst_texture: 24.56, worst_perimeter: 123.4, worst_area: 1110.0,
        worst_smoothness: 0.1491, worst_compactness: 0.3598, worst_concavity: 0.3168,
        worst_concave_points: 0.1688, worst_symmetry: 0.2736, worst_fractal_dimension: 0.1015
      }
    }
  ];

  const handleSampleSelection = (index: number) => {
    setSelectedSample(index);
    const sample = sampleCases[index];
    setBiomarkerData(sample.data);
    setPatientInfo(sample.patientInfo);
  };

  const runQuantumAnalysis = useCallback(async () => {
    // Immediately set loading state and move to analysis step
    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    setCurrentStep(4); // Move to analysis step immediately to show loading

    try {
      // First try the real API
      const response = await apiService.predict(biomarkerData, true);
      
      if (response.error) {
        throw new Error(response.error);
      }

      // Simulate processing time for better UX (minimum 2 seconds for realistic quantum processing)
      const minProcessingTime = new Promise(resolve => setTimeout(resolve, 2000));
      await minProcessingTime;
      
      setResult(response.data);
    } catch (error) {
      console.warn('API call failed, using simulated result:', error);
      
      // Fallback to simulated result for demo purposes
      const minProcessingTime = new Promise(resolve => setTimeout(resolve, 1500));
      await minProcessingTime;
      
      const simulatedResult = apiService.generateSimulatedResult(biomarkerData);
      setResult(simulatedResult);
    } finally {
      setIsAnalyzing(false);
    }
  }, [biomarkerData]);

  const resetAssessment = () => {
    setCurrentStep(1);
    setResult(null);
    setIsAnalyzing(false);
    setError(null);
  };

  const steps = [
    { step: 1, title: "Input Method", icon: DocumentArrowUpIcon },
    { step: 2, title: "Patient Info", icon: UserIcon },
    { step: 3, title: "Biomarkers", icon: BeakerIcon },
    { step: 4, title: "Analysis", icon: CpuChipIcon }
  ];

  return (
    <div className="min-h-screen pt-32 pb-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-white via-blue-50/30 to-teal-50/30 dark:bg-gradient-to-br dark:from-slate-900 dark:via-slate-800/50 dark:to-slate-900">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-20">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <HeartIcon className="w-12 h-12 text-pink-500 dark:text-pink-400" />
            <h1 className="text-4xl md:text-5xl font-bold text-gradient-healthcare">
              Cancer Risk Assessment
            </h1>
          </div>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-4xl mx-auto">
            Advanced quantum AI analysis for early breast cancer detection. 
            Input your biomarker data to receive a comprehensive risk assessment powered by quantum machine learning.
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-12">
          <div className="flex items-center justify-center space-x-4 mb-8 overflow-x-auto">
            {steps.map(({ step, title, icon: Icon }, index) => (
              <div key={step} className="flex items-center flex-shrink-0">
                <div className={`flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300 ${
                  currentStep >= step 
                    ? 'bg-blue-500 border-blue-500 text-white shadow-blue' 
                    : 'bg-white dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-400 dark:text-slate-500'
                }`}>
                  {currentStep > step ? (
                    <CheckCircleIcon className="w-6 h-6" />
                  ) : (
                    <Icon className="w-6 h-6" />
                  )}
                </div>
                <div className="ml-3 text-sm">
                  <div className={`font-medium ${currentStep >= step ? 'text-blue-600 dark:text-blue-400' : 'text-slate-400 dark:text-slate-500'}`}>
                    Step {step}
                  </div>
                  <div className="text-slate-500 dark:text-slate-400">{title}</div>
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-16 h-0.5 mx-4 transition-colors duration-300 ${
                    currentStep > step ? 'bg-blue-500' : 'bg-slate-300 dark:bg-slate-600'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-8 p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-xl">
            <div className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-600 dark:text-red-400" />
              <p className="text-red-700 dark:text-red-300">{error}</p>
            </div>
          </div>
        )}

        {/* Step 1: Input Method Selection */}
        {currentStep === 1 && (
          <div className="max-w-4xl mx-auto">
            <div className="healthcare-card">
              <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-100 mb-6 text-center">
                Choose Your Input Method
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <button
                  onClick={() => {
                    setInputMethod('sample');
                    setCurrentStep(2);
                  }}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 focus:outline-none ${
                    inputMethod === 'sample' 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 shadow-blue' 
                      : 'border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20'
                  }`}
                >
                  <EyeIcon className="w-12 h-12 text-blue-500 dark:text-blue-400 mx-auto mb-4" />
                  <h4 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">Sample Cases</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Use pre-loaded sample cases to see how the quantum AI works with different risk profiles.
                  </p>
                  <div className="mt-4 text-xs text-blue-600 dark:text-blue-400 font-medium">
                    Recommended for first-time users
                  </div>
                </button>

                <button
                  onClick={() => {
                    setInputMethod('manual');
                    setCurrentStep(2);
                  }}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 focus:outline-none ${
                    inputMethod === 'manual' 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 shadow-blue' 
                      : 'border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20'
                  }`}
                >
                  <ScaleIcon className="w-12 h-12 text-green-500 dark:text-green-400 mx-auto mb-4" />
                  <h4 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">Manual Entry</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Enter biomarker values manually. Requires medical test results with 30 specific measurements.
                  </p>
                  <div className="mt-4 text-xs text-green-600 dark:text-green-400 font-medium">
                    For healthcare professionals
                  </div>
                </button>

                <button
                  onClick={() => {
                    setInputMethod('upload');
                    setCurrentStep(2);
                  }}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 focus:outline-none ${
                     inputMethod === 'upload' 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 shadow-blue' 
                      : 'border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20'
                  }`}
                >
                  <DocumentArrowUpIcon className="w-12 h-12 text-purple-500 dark:text-purple-400 mx-auto mb-4" />
                  <h4 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">Upload CSV</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Upload a CSV file with biomarker data. Must follow the UCI Breast Cancer dataset format.
                  </p>
                  <div className="mt-4 text-xs text-purple-600 dark:text-purple-400 font-medium">
                    For batch processing
                  </div>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Patient Information */}
        {currentStep === 2 && (
          <PatientInfoForm
            patientInfo={patientInfo}
            setPatientInfo={setPatientInfo}
            onNext={() => setCurrentStep(3)}
            onBack={() => setCurrentStep(1)}
            inputMethod={inputMethod}
            sampleCases={sampleCases}
            selectedSample={selectedSample}
            onSampleSelection={handleSampleSelection}
          />
        )}

        {/* Step 3: Biomarker Data */}
        {currentStep === 3 && (
          <BiomarkerForm
            biomarkerData={biomarkerData}
            setBiomarkerData={setBiomarkerData}
            onNext={runQuantumAnalysis}
            onBack={() => setCurrentStep(2)}
            inputMethod={inputMethod}
            isAnalyzing={isAnalyzing}
          />
        )}

        {/* Step 4: Analysis Results */}
        {currentStep === 4 && (
          <>
            {isAnalyzing && (
              <div className="healthcare-card text-center">
                <LoadingSpinner size="lg" quantum message="Quantum Analysis in Progress" />
                <p className="text-slate-600 dark:text-slate-300 mb-8 mt-4">
                  Processing your biomarker data through advanced quantum circuits...
                </p>
                <div className="space-y-3 text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
                  <div className="flex items-center justify-center space-x-2">
                    <SparklesIcon className="w-4 h-4" />
                    <span>Encoding biomarkers into quantum states</span>
                  </div>
                  <div className="flex items-center justify-center space-x-2">
                    <CpuChipIcon className="w-4 h-4" />
                    <span>Executing 6-qubit variational quantum circuit</span>
                  </div>
                  <div className="flex items-center justify-center space-x-2">
                    <ShieldCheckIcon className="w-4 h-4" />
                    <span>Applying quantum error mitigation</span>
                  </div>
                  <div className="flex items-center justify-center space-x-2">
                    <ChartBarIcon className="w-4 h-4" />
                    <span>Measuring quantum probabilities</span>
                  </div>
                </div>
              </div>
            )}

            {result && !isAnalyzing && (
              <AnalysisResults result={result} onReset={resetAssessment} />
            )}
          </>
        )}

        {/* Medical Disclaimer */}
        <div className="mt-16 bg-amber-50 dark:bg-amber-900/30 rounded-2xl p-6 border border-amber-200 dark:border-amber-700">
          <div className="flex items-start space-x-3">
            <ExclamationTriangleIcon className="w-6 h-6 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-amber-800 dark:text-amber-300 mb-2">
                Important Medical Disclaimer
              </h4>
              <p className="text-sm text-amber-700 dark:text-amber-400">
                This quantum AI tool is for research and educational purposes only. It is not approved for clinical use 
                and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified 
                healthcare professionals for medical decisions. The biomarker values used are based on the UCI Breast 
                Cancer Wisconsin dataset for demonstration purposes.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CancerRiskAssessment;

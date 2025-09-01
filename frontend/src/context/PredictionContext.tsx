import React, { createContext, useContext, useState, ReactNode } from 'react';

export interface PatientData {
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

export interface PredictionResult {
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
    measurement_counts: Record<string, number>;
  };
}

interface PredictionContextType {
  patientData: PatientData | null;
  predictionResult: PredictionResult | null;
  isLoading: boolean;
  error: string | null;
  setPatientData: (data: PatientData) => void;
  setPredictionResult: (result: PredictionResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearData: () => void;
}

const PredictionContext = createContext<PredictionContextType | undefined>(undefined);

export const usePrediction = () => {
  const context = useContext(PredictionContext);
  if (context === undefined) {
    throw new Error('usePrediction must be used within a PredictionProvider');
  }
  return context;
};

interface PredictionProviderProps {
  children: ReactNode;
}

export const PredictionProvider: React.FC<PredictionProviderProps> = ({ children }) => {
  const [patientData, setPatientData] = useState<PatientData | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setLoading = (loading: boolean) => {
    setIsLoading(loading);
    if (loading) {
      setError(null);
    }
  };

  const clearData = () => {
    setPatientData(null);
    setPredictionResult(null);
    setError(null);
    setIsLoading(false);
  };

  const value: PredictionContextType = {
    patientData,
    predictionResult,
    isLoading,
    error,
    setPatientData,
    setPredictionResult,
    setLoading,
    setError,
    clearData,
  };

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
};
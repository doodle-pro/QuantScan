import React, { useState, useEffect } from 'react';
import { 
  CpuChipIcon, 
  PlayIcon, 
  PauseIcon, 
  ArrowPathIcon,
  ChartBarIcon,
  BeakerIcon,
  LightBulbIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

const QuantumDemo = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [quantumState, setQuantumState] = useState([1, 0, 0, 0, 0, 0, 0, 0]);
  const [measurementResults, setMeasurementResults] = useState({ '0': 0, '1': 0 });

  const circuitSteps = [
    {
      name: 'Initialize',
      description: 'Start with all qubits in |0⟩ state',
      code: 'qubits = |000000⟩',
      color: 'from-slate-400 to-slate-500'
    },
    {
      name: 'Feature Encoding',
      description: 'Encode medical data using RY rotations',
      code: 'RY(θ_feature) → qubits',
      color: 'from-blue-500 to-blue-600'
    },
    {
      name: 'Superposition',
      description: 'Create quantum superposition with Hadamard gates',
      code: 'H → qubits',
      color: 'from-teal-500 to-teal-600'
    },
    {
      name: 'Entanglement',
      description: 'Create quantum entanglement between qubits',
      code: 'CNOT → qubits',
      color: 'from-green-500 to-green-600'
    },
    {
      name: 'Variational Layer',
      description: 'Apply trainable quantum gates',
      code: 'RY(θ_train), RZ(φ_train) → qubits',
      color: 'from-success-500 to-success-600'
    },
    {
      name: 'Measurement',
      description: 'Measure first qubit for classification',
      code: 'M → qubit[0]',
      color: 'from-pink-400 to-pink-500'
    }
  ];

  const quantumGates = [
    { name: 'RY', symbol: 'RY(θ)', description: 'Y-axis rotation for feature encoding' },
    { name: 'RZ', symbol: 'RZ(φ)', description: 'Z-axis rotation for phase encoding' },
    { name: 'H', symbol: 'H', description: 'Hadamard gate for superposition' },
    { name: 'CNOT', symbol: '⊕', description: 'Controlled-NOT for entanglement' },
    { name: 'CZ', symbol: 'CZ', description: 'Controlled-Z for correlation' }
  ];

  const medicalFeatures = [
    { name: 'Radius Mean', value: 14.127, encoded: 0.785, unit: 'mm' },
    { name: 'Texture Mean', value: 19.289, encoded: 1.204, unit: 'std' },
    { name: 'Perimeter Mean', value: 91.969, encoded: 0.573, unit: 'mm' },
    { name: 'Area Mean', value: 654.889, encoded: 0.408, unit: 'mm²' },
    { name: 'Smoothness Mean', value: 0.096, encoded: 0.301, unit: 'ratio' },
    { name: 'Compactness Mean', value: 0.104, encoded: 0.650, unit: 'ratio' }
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRunning) {
      interval = setInterval(() => {
        setCurrentStep((prev) => {
          const next = (prev + 1) % circuitSteps.length;
          if (next === 0) {
            // Reset quantum state
            setQuantumState([1, 0, 0, 0, 0, 0, 0, 0]);
            setMeasurementResults({ '0': 0, '1': 0 });
          } else if (next === circuitSteps.length - 1) {
            // Simulate measurement
            const prob1 = Math.random() * 0.4 + 0.3; // 30-70% probability
            const shots = 1000;
            const count1 = Math.floor(prob1 * shots);
            const count0 = shots - count1;
            setMeasurementResults({ '0': count0, '1': count1 });
          }
          return next;
        });
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const toggleSimulation = () => {
    setIsRunning(!isRunning);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setCurrentStep(0);
    setQuantumState([1, 0, 0, 0, 0, 0, 0, 0]);
    setMeasurementResults({ '0': 0, '1': 0 });
  };

  const renderQuantumCircuit = () => (
    <div className="healthcare-card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
          Quantum Circuit Visualization
        </h3>
        <div className="flex space-x-2">
          <button
            onClick={toggleSimulation}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
              isRunning
                ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/50 border border-red-200 dark:border-red-700'
                : 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 hover:bg-green-100 dark:hover:bg-green-900/50 border border-green-200 dark:border-green-700'
            }`}
          >
            {isRunning ? <PauseIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
            <span>{isRunning ? 'Pause' : 'Run'}</span>
          </button>
          <button
            onClick={resetSimulation}
            className="btn-soft flex items-center space-x-2"
          >
            <ArrowPathIcon className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Circuit Diagram */}
      <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-6 mb-6 overflow-x-auto">
        <div className="min-w-[800px]">
          {/* Qubit Lines */}
          {[0, 1, 2, 3, 4, 5].map((qubit) => (
            <div key={qubit} className="flex items-center mb-4 last:mb-0">
              <div className="w-16 text-sm font-mono text-slate-600 dark:text-slate-300">
                q{qubit}: |0⟩
              </div>
              <div className="flex-1 h-0.5 bg-slate-300 dark:bg-slate-600 relative">
                {/* Gates */}
                {circuitSteps.map((step, stepIndex) => (
                  <div
                    key={stepIndex}
                    className={`absolute top-1/2 transform -translate-y-1/2 w-12 h-8 rounded-lg border-2 flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                      stepIndex <= currentStep
                        ? `bg-gradient-to-r ${step.color} text-white border-transparent shadow-comfortable`
                        : 'bg-white dark:bg-slate-700 text-slate-400 dark:text-slate-500 border-slate-300 dark:border-slate-600'
                    }`}
                    style={{ left: `${stepIndex * 120 + 20}px` }}
                  >
                    {stepIndex === 0 && '|0⟩'}
                    {stepIndex === 1 && 'RY'}
                    {stepIndex === 2 && 'H'}
                    {stepIndex === 3 && (qubit < 5 ? '●' : '⊕')}
                    {stepIndex === 4 && 'RY'}
                    {stepIndex === 5 && qubit === 0 && 'M'}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Current Step Info */}
      <div className={`bg-gradient-to-r ${circuitSteps[currentStep].color} rounded-xl p-4 text-white shadow-comfortable`}>
        <div className="flex items-center space-x-3 mb-2">
          <SparklesIcon className="w-5 h-5" />
          <h4 className="font-semibold">Step {currentStep + 1}: {circuitSteps[currentStep].name}</h4>
        </div>
        <p className="text-sm opacity-90 mb-2">{circuitSteps[currentStep].description}</p>
        <code className="text-xs bg-black/20 px-2 py-1 rounded font-mono">
          {circuitSteps[currentStep].code}
        </code>
      </div>
    </div>
  );

  const renderMeasurementResults = () => (
    <div className="healthcare-card">
      <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-6">
        Measurement Results
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Probability Distribution */}
        <div>
          <h4 className="text-lg font-medium text-slate-700 dark:text-slate-200 mb-4">
            Probability Distribution
          </h4>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-600 dark:text-slate-300">|0⟩ (Benign)</span>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">
                  {((measurementResults['0'] / (measurementResults['0'] + measurementResults['1']) || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-500"
                  style={{ 
                    width: `${(measurementResults['0'] / (measurementResults['0'] + measurementResults['1']) || 0) * 100}%` 
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-600 dark:text-slate-300">|1⟩ (Malignant)</span>
                <span className="text-sm font-medium text-red-600 dark:text-red-400">
                  {((measurementResults['1'] / (measurementResults['0'] + measurementResults['1']) || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-red-500 to-red-600 h-3 rounded-full transition-all duration-500"
                  style={{ 
                    width: `${(measurementResults['1'] / (measurementResults['0'] + measurementResults['1']) || 0) * 100}%` 
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Shot Counts */}
        <div>
          <h4 className="text-lg font-medium text-slate-700 dark:text-slate-200 mb-4">
            Shot Counts (1000 shots)
          </h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/30 rounded-xl border border-green-200 dark:border-green-700">
              <span className="text-sm font-medium text-green-700 dark:text-green-300">Benign (|0⟩)</span>
              <span className="text-lg font-bold text-green-600 dark:text-green-400">{measurementResults['0']}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/30 rounded-xl border border-red-200 dark:border-red-700">
              <span className="text-sm font-medium text-red-700 dark:text-red-300">Malignant (|1⟩)</span>
              <span className="text-lg font-bold text-red-600 dark:text-red-400">{measurementResults['1']}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl border border-blue-200 dark:border-blue-700">
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Total Shots</span>
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {measurementResults['0'] + measurementResults['1']}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderFeatureEncoding = () => (
    <div className="healthcare-card">
      <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-6">
        Medical Feature Encoding
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-medium text-slate-700 dark:text-slate-200 mb-4">
            UCI Dataset Features
          </h4>
          <div className="space-y-3">
            {medicalFeatures.map((feature, index) => (
              <div key={index} className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
                <div>
                  <div className="text-sm font-medium text-slate-700 dark:text-slate-200">
                    {feature.name}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    {feature.value} {feature.unit}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold text-blue-600 dark:text-blue-400">
                    θ = {feature.encoded.toFixed(3)}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    radians
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-lg font-medium text-slate-700 dark:text-slate-200 mb-4">
            Quantum Gates Reference
          </h4>
          <div className="space-y-3">
            {quantumGates.map((gate, index) => {
              const colors = [
                'from-blue-500 to-teal-600',
                'from-teal-500 to-green-600',
                'from-green-500 to-blue-600',
                'from-green-500 to-blue-600',
                'from-blue-500 to-green-600'
              ];
              return (
                <div key={index} className="flex items-center space-x-4 p-3 bg-teal-50 dark:bg-teal-900/30 rounded-xl">
                  <div className={`w-12 h-8 bg-gradient-to-r ${colors[index]} rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-comfortable`}>
                    {gate.symbol}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-slate-700 dark:text-slate-200">
                      {gate.name}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {gate.description}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen pt-32 pb-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-white via-blue-50/30 to-teal-50/30 dark:bg-gradient-to-br dark:from-slate-900 dark:via-slate-800/50 dark:to-slate-900">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-20">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <CpuChipIcon className="w-12 h-12 text-blue-600 dark:text-blue-400" />
            <h1 className="text-4xl md:text-5xl font-bold text-gradient-healthcare">
              Quantum Circuit Demo
            </h1>
          </div>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            Interactive visualization of Q-MediScan's quantum machine learning circuit. 
            Watch how medical data flows through quantum gates to produce cancer detection results.
          </p>
        </div>

        {/* Demo Sections */}
        <div className="space-y-8">
          {/* Quantum Circuit */}
          {renderQuantumCircuit()}

          {/* Feature Encoding */}
          {renderFeatureEncoding()}

          {/* Measurement Results */}
          {renderMeasurementResults()}

          {/* Technical Details */}
          <div className="bg-teal-50 dark:bg-teal-900/30 rounded-2xl p-8 border border-teal-200 dark:border-teal-700 shadow-teal">
            <div className="flex items-center space-x-3 mb-6">
              <BeakerIcon className="w-8 h-8 text-teal-600 dark:text-teal-400" />
              <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                Technical Implementation
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-teal-600 dark:text-teal-400 mb-2">Classiq SDK</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Quantum Framework</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">NISQ</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Hardware Compatible</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">VQC</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Variational Quantum Classifier</div>
              </div>
            </div>
            <div className="mt-6 p-4 bg-white/50 dark:bg-slate-800/50 rounded-xl">
              <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-2">Key Features:</h4>
              <ul className="text-sm text-slate-600 dark:text-slate-300 space-y-1">
                <li>• Amplitude encoding for efficient feature representation</li>
                <li>• Hardware-efficient ansatz optimized for NISQ devices</li>
                <li>• Error mitigation techniques for reliable medical predictions</li>
                <li>• Real-time quantum state visualization and measurement</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuantumDemo;

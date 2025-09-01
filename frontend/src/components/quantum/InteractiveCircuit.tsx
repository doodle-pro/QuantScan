import React, { useState, useEffect } from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  ArrowPathIcon,
  SparklesIcon,
  CpuChipIcon,
  BeakerIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import Button from '../ui/Button';
import Tooltip from '../ui/Tooltip';

interface CircuitStep {
  name: string;
  description: string;
  code: string;
  color: string;
  duration: number;
}

interface QuantumGate {
  name: string;
  symbol: string;
  description: string;
  color: string;
  interactive: boolean;
}

interface InteractiveCircuitProps {
  onStateChange?: (state: number[]) => void;
  onMeasurement?: (results: { [key: string]: number }) => void;
}

const InteractiveCircuit: React.FC<InteractiveCircuitProps> = ({
  onStateChange,
  onMeasurement
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [quantumState, setQuantumState] = useState([1, 0, 0, 0, 0, 0, 0, 0]);
  const [measurementResults, setMeasurementResults] = useState({ '0': 0, '1': 0 });
  const [selectedGate, setSelectedGate] = useState<string | null>(null);
  const [gateParameters, setGateParameters] = useState<{ [key: string]: number }>({
    theta: Math.PI / 4,
    phi: Math.PI / 2
  });

  const circuitSteps: CircuitStep[] = [
    {
      name: 'Initialize',
      description: 'Start with all qubits in |0⟩ state',
      code: 'qubits = |000000⟩',
      color: 'from-slate-400 to-slate-500',
      duration: 1000
    },
    {
      name: 'Feature Encoding',
      description: 'Encode medical data using RY rotations',
      code: `RY(${gateParameters.theta.toFixed(3)}) → qubits`,
      color: 'from-blue-500 to-blue-600',
      duration: 1500
    },
    {
      name: 'Superposition',
      description: 'Create quantum superposition with Hadamard gates',
      code: 'H → qubits',
      color: 'from-teal-500 to-teal-600',
      duration: 1200
    },
    {
      name: 'Entanglement',
      description: 'Create quantum entanglement between qubits',
      code: 'CNOT → qubits',
      color: 'from-green-500 to-green-600',
      duration: 1800
    },
    {
      name: 'Variational Layer',
      description: 'Apply trainable quantum gates',
      code: `RY(${gateParameters.theta.toFixed(3)}), RZ(${gateParameters.phi.toFixed(3)}) → qubits`,
      color: 'from-purple-500 to-purple-600',
      duration: 2000
    },
    {
      name: 'Measurement',
      description: 'Measure first qubit for classification',
      code: 'M → qubit[0]',
      color: 'from-pink-400 to-pink-500',
      duration: 1000
    }
  ];

  const quantumGates: QuantumGate[] = [
    { name: 'RY', symbol: 'RY(θ)', description: 'Y-axis rotation for feature encoding', color: 'bg-blue-500', interactive: true },
    { name: 'RZ', symbol: 'RZ(φ)', description: 'Z-axis rotation for phase encoding', color: 'bg-teal-500', interactive: true },
    { name: 'H', symbol: 'H', description: 'Hadamard gate for superposition', color: 'bg-green-500', interactive: false },
    { name: 'CNOT', symbol: '⊕', description: 'Controlled-NOT for entanglement', color: 'bg-purple-500', interactive: false },
    { name: 'CZ', symbol: 'CZ', description: 'Controlled-Z for correlation', color: 'bg-pink-500', interactive: false }
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRunning) {
      interval = setInterval(() => {
        setCurrentStep((prev) => {
          const next = (prev + 1) % circuitSteps.length;
          if (next === 0) {
            // Reset quantum state
            const newState = [1, 0, 0, 0, 0, 0, 0, 0];
            setQuantumState(newState);
            setMeasurementResults({ '0': 0, '1': 0 });
            onStateChange?.(newState);
          } else if (next === circuitSteps.length - 1) {
            // Simulate measurement
            const prob1 = Math.random() * 0.4 + 0.3; // 30-70% probability
            const shots = 1000;
            const count1 = Math.floor(prob1 * shots);
            const count0 = shots - count1;
            const results = { '0': count0, '1': count1 };
            setMeasurementResults(results);
            onMeasurement?.(results);
          }
          return next;
        });
      }, circuitSteps[currentStep]?.duration || 2000);
    }
    return () => clearInterval(interval);
  }, [isRunning, currentStep, circuitSteps, onStateChange, onMeasurement]);

  const toggleSimulation = () => {
    setIsRunning(!isRunning);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setCurrentStep(0);
    const initialState = [1, 0, 0, 0, 0, 0, 0, 0];
    setQuantumState(initialState);
    setMeasurementResults({ '0': 0, '1': 0 });
    onStateChange?.(initialState);
  };

  const handleGateClick = (gateName: string) => {
    if (quantumGates.find(g => g.name === gateName)?.interactive) {
      setSelectedGate(selectedGate === gateName ? null : gateName);
    }
  };

  const handleParameterChange = (param: string, value: number) => {
    setGateParameters(prev => ({ ...prev, [param]: value }));
  };

  const stepToNextManually = () => {
    if (!isRunning) {
      setCurrentStep((prev) => (prev + 1) % circuitSteps.length);
    }
  };

  return (
    <div className="healthcare-card">
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6 gap-4">
        <div>
          <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-2">
            Interactive Quantum Circuit
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-300">
            Click gates to adjust parameters, or run the automatic simulation
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant={isRunning ? 'danger' : 'primary'}
            size="sm"
            onClick={toggleSimulation}
            leftIcon={isRunning ? <PauseIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
          >
            {isRunning ? 'Pause' : 'Run Auto'}
          </Button>
          <Button
            variant="soft"
            size="sm"
            onClick={stepToNextManually}
            disabled={isRunning}
            leftIcon={<SparklesIcon className="w-4 h-4" />}
          >
            Step
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={resetSimulation}
            leftIcon={<ArrowPathIcon className="w-4 h-4" />}
          >
            Reset
          </Button>
        </div>
      </div>

      {/* Circuit Visualization */}
      <div className="bg-gradient-to-br from-blue-50 to-teal-50 dark:from-blue-900/30 dark:to-teal-900/30 rounded-xl p-6 mb-6 overflow-x-auto">
        <div className="min-w-[900px]">
          {/* Qubit Lines */}
          {[0, 1, 2, 3, 4, 5].map((qubit) => (
            <div key={qubit} className="flex items-center mb-4 last:mb-0">
              <div className="w-20 text-sm font-mono text-slate-600 dark:text-slate-300 flex items-center">
                <span>q{qubit}: |0⟩</span>
                <Tooltip content={`Qubit ${qubit} - starts in ground state |0⟩`}>
                  <div className="ml-2 w-2 h-2 bg-blue-400 rounded-full"></div>
                </Tooltip>
              </div>
              <div className="flex-1 h-0.5 bg-slate-300 dark:bg-slate-600 relative">
                {/* Gates */}
                {circuitSteps.map((step, stepIndex) => {
                  const isActive = stepIndex <= currentStep;
                  const isCurrent = stepIndex === currentStep;
                  
                  return (
                    <Tooltip
                      key={stepIndex}
                      content={`${step.name}: ${step.description}`}
                    >
                      <button
                        className={`absolute top-1/2 transform -translate-y-1/2 w-14 h-10 rounded-lg border-2 flex items-center justify-center text-xs font-bold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-500 ${
                          isActive
                            ? `bg-gradient-to-r ${step.color} text-white border-transparent shadow-lg ${
                                isCurrent ? 'scale-110 animate-pulse' : ''
                              }`
                            : 'bg-white dark:bg-slate-700 text-slate-400 dark:text-slate-500 border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500'
                        }`}
                        style={{ left: `${stepIndex * 140 + 20}px` }}
                        onClick={() => handleGateClick(step.name)}
                        disabled={isRunning}
                      >
                        {stepIndex === 0 && '|0⟩'}
                        {stepIndex === 1 && 'RY'}
                        {stepIndex === 2 && 'H'}
                        {stepIndex === 3 && (qubit < 5 ? '●' : '⊕')}
                        {stepIndex === 4 && 'RY'}
                        {stepIndex === 5 && qubit === 0 && 'M'}
                      </button>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Current Step Info */}
      <div className={`bg-gradient-to-r ${circuitSteps[currentStep].color} rounded-xl p-6 text-white shadow-lg mb-6`}>
        <div className="flex items-center space-x-3 mb-3">
          <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
            <span className="text-sm font-bold">{currentStep + 1}</span>
          </div>
          <h4 className="font-semibold text-lg">
            {circuitSteps[currentStep].name}
          </h4>
        </div>
        <p className="text-sm opacity-90 mb-3">{circuitSteps[currentStep].description}</p>
        <div className="bg-black/20 rounded-lg p-3">
          <code className="text-xs font-mono">
            {circuitSteps[currentStep].code}
          </code>
        </div>
      </div>

      {/* Interactive Gate Parameters */}
      {selectedGate && (
        <div className="bg-slate-50 dark:bg-slate-700 rounded-xl p-6 mb-6 border-2 border-blue-200 dark:border-blue-600">
          <h4 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">
            Adjust {selectedGate} Gate Parameters
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {selectedGate === 'RY' && (
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
                  Theta (θ) - Rotation Angle
                </label>
                <input
                  type="range"
                  min="0"
                  max={Math.PI * 2}
                  step="0.1"
                  value={gateParameters.theta}
                  onChange={(e) => handleParameterChange('theta', parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {gateParameters.theta.toFixed(3)} radians ({(gateParameters.theta * 180 / Math.PI).toFixed(1)}°)
                </div>
              </div>
            )}
            {selectedGate === 'RZ' && (
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
                  Phi (φ) - Phase Angle
                </label>
                <input
                  type="range"
                  min="0"
                  max={Math.PI * 2}
                  step="0.1"
                  value={gateParameters.phi}
                  onChange={(e) => handleParameterChange('phi', parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {gateParameters.phi.toFixed(3)} radians ({(gateParameters.phi * 180 / Math.PI).toFixed(1)}°)
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Quantum Gates Reference */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {quantumGates.map((gate, index) => (
          <Tooltip key={index} content={gate.description}>
            <button
              onClick={() => handleGateClick(gate.name)}
              className={`p-4 rounded-xl border-2 transition-all duration-200 text-left hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-500 ${
                selectedGate === gate.name
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                  : gate.interactive
                  ? 'border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20'
                  : 'border-slate-200 dark:border-slate-600 cursor-default'
              }`}
              disabled={!gate.interactive}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-10 h-8 ${gate.color} rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm`}>
                  {gate.symbol}
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-700 dark:text-slate-200">
                    {gate.name}
                    {gate.interactive && (
                      <span className="ml-2 text-xs text-blue-600 dark:text-blue-400">Interactive</span>
                    )}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    {gate.description}
                  </div>
                </div>
              </div>
            </button>
          </Tooltip>
        ))}
      </div>
    </div>
  );
};

export default InteractiveCircuit;
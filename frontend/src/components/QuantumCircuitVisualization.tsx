import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Zap, Activity, Info } from 'lucide-react';

interface CircuitInfo {
  qubits: number;
  layers: number;
  parameters: number;
  measurement_counts: Record<string, number>;
}

interface QuantumCircuitVisualizationProps {
  circuitInfo?: CircuitInfo;
}

const QuantumCircuitVisualization: React.FC<QuantumCircuitVisualizationProps> = ({ 
  circuitInfo 
}) => {
  const qubits = circuitInfo?.qubits || 6;
  const layers = circuitInfo?.layers || 3;
  
  // Generate quantum circuit diagram
  const generateCircuitDiagram = () => {
    const lines = [];
    
    // Header
    lines.push('Quantum Breast Cancer Classifier Circuit');
    lines.push('═'.repeat(60));
    lines.push('');
    
    // Circuit representation
    for (let q = 0; q < qubits; q++) {
      let line = `q${q}: `;
      
      // Initial state
      line += '|0⟩─';
      
      // Feature encoding layer
      line += `RY(θ${q})─`;
      
      // Variational layers
      for (let layer = 0; layer < layers; layer++) {
        line += `RY(α${layer}${q})─RZ(β${layer}${q})─`;
        
        // Entangling gates
        if (q < qubits - 1) {
          line += '●─';
        } else if (q === qubits - 1 && layer < layers - 1) {
          line += '│─';
        }
      }
      
      // Measurement
      if (q === 0) {
        line += '─M─';
      } else {
        line += '─│─';
      }
      
      lines.push(line);
      
      // CNOT connections
      if (q < qubits - 1) {
        let connection = ' '.repeat(line.length - 10);
        connection += '│';
        lines.push(connection);
      }
    }
    
    lines.push('');
    lines.push('Classical Output: ═══════════════════════════════════════');
    lines.push('');
    lines.push('Legend:');
    lines.push('RY(θ) - Rotation Y gate (feature encoding)');
    lines.push('RY(α), RZ(β) - Variational parameters');
    lines.push('● - Control qubit, │ - Target connection');
    lines.push('M - Measurement operation');
    
    return lines.join('\n');
  };

  const circuitDiagram = generateCircuitDiagram();

  // Generate gate sequence visualization
  const generateGateSequence = () => {
    const gates = [];
    
    // Feature encoding gates
    for (let q = 0; q < qubits; q++) {
      gates.push({
        type: 'RY',
        qubit: q,
        parameter: `θ${q}`,
        description: 'Feature encoding',
        color: 'bg-blue-500'
      });
    }
    
    // Variational layers
    for (let layer = 0; layer < layers; layer++) {
      for (let q = 0; q < qubits; q++) {
        gates.push({
          type: 'RY',
          qubit: q,
          parameter: `α${layer}${q}`,
          description: `Layer ${layer + 1} rotation`,
          color: 'bg-teal-500'
        });
        gates.push({
          type: 'RZ',
          qubit: q,
          parameter: `β${layer}${q}`,
          description: `Layer ${layer + 1} phase`,
          color: 'bg-green-500'
        });
      }
      
      // Entangling gates
      for (let q = 0; q < qubits - 1; q++) {
        gates.push({
          type: 'CNOT',
          qubit: `${q},${q + 1}`,
          parameter: '',
          description: 'Entangling gate',
          color: 'bg-green-500'
        });
      }
    }
    
    return gates;
  };

  const gateSequence = generateGateSequence();

  return (
    <div className="space-y-8">
      {/* Circuit Overview */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
            <Cpu className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Quantum Circuit Architecture
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Variational Quantum Classifier for biomarker analysis
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Circuit Diagram */}
          <div>
            <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-4">Circuit Diagram</h4>
            <div className="bg-slate-900 dark:bg-slate-800 rounded-xl p-4 overflow-x-auto">
              <pre className="text-green-400 dark:text-green-300 text-xs leading-relaxed">
                {circuitDiagram}
              </pre>
            </div>
          </div>
          
          {/* Circuit Statistics */}
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-4">Circuit Statistics</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{qubits}</div>
                  <div className="text-sm text-blue-700 dark:text-blue-400">Qubits</div>
                </div>
                <div className="bg-teal-50 dark:bg-teal-900/30 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold text-teal-600 dark:text-teal-400">{layers}</div>
                  <div className="text-sm text-teal-700 dark:text-teal-400">Layers</div>
                </div>
                <div className="bg-pink-50 dark:bg-pink-900/30 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold text-pink-600 dark:text-pink-400">
                    {circuitInfo?.parameters || layers * 3 * qubits}
                  </div>
                  <div className="text-sm text-pink-700 dark:text-pink-400">Parameters</div>
                </div>
                <div className="bg-green-50 dark:bg-green-900/30 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {qubits - 1}
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-400">CNOT Gates</div>
                </div>
              </div>
            </div>
            
            {/* Circuit Depth */}
            <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4">
              <h5 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Circuit Depth Analysis</h5>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-blue-600 dark:text-blue-400">Feature Encoding</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">1 layer</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-600 dark:text-blue-400">Variational Layers</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{layers} layers</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-600 dark:text-blue-400">Total Depth</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{1 + layers * 2} gates</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Gate Sequence */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-teal-100 dark:bg-teal-900/50 rounded-full flex items-center justify-center">
            <Activity className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Gate Sequence Visualization
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Step-by-step quantum operations
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {gateSequence.slice(0, 20).map((gate, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white dark:bg-slate-700 border border-blue-200 dark:border-blue-600 rounded-xl p-3 hover:shadow-comfortable transition-shadow duration-200"
            >
              <div className="flex items-center space-x-2 mb-2">
                <div className={`w-6 h-6 ${gate.color} text-white rounded text-xs font-bold flex items-center justify-center`}>
                  {gate.type === 'CNOT' ? '⊕' : gate.type}
                </div>
                <div className="text-sm font-medium text-slate-800 dark:text-slate-100">
                  q{gate.qubit}
                </div>
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-300 mb-1">
                {gate.parameter && `Parameter: ${gate.parameter}`}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                {gate.description}
              </div>
            </motion.div>
          ))}
          
          {gateSequence.length > 20 && (
            <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-600 rounded-xl p-3 flex items-center justify-center">
              <div className="text-center text-blue-600 dark:text-blue-400">
                <div className="text-lg font-bold">+{gateSequence.length - 20}</div>
                <div className="text-xs">more gates</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Measurement Results */}
      {circuitInfo?.measurement_counts && (
        <div className="healthcare-card">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center">
              <Zap className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
                Quantum Measurement Results
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Quantum state collapse outcomes
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(circuitInfo.measurement_counts).map(([state, count]) => {
              const total = Object.values(circuitInfo.measurement_counts).reduce((a, b) => a + b, 0);
              const percentage = (count / total) * 100;
              
              return (
                <div key={state} className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4 text-center">
                  <div className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-1">
                    |{state}⟩
                  </div>
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                    {count}
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-300">
                    {percentage.toFixed(1)}%
                  </div>
                  <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-2 mt-2">
                    <div 
                      className="bg-blue-500 dark:bg-blue-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="mt-6 bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4">
            <div className="flex items-start space-x-3">
              <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-1">Measurement Interpretation</h4>
                <p className="text-blue-700 dark:text-blue-400 text-sm">
                  The quantum circuit was executed with {Object.values(circuitInfo.measurement_counts).reduce((a, b) => a + b, 0)} shots. 
                  The measurement results show the probability distribution of quantum states, 
                  where |1⟩ states indicate higher cancer risk probability and |0⟩ states indicate lower risk.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quantum Advantage Explanation */}
      <div className="bg-teal-50 dark:bg-teal-900/30 rounded-2xl p-6 border border-teal-200 dark:border-teal-700 shadow-teal">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-teal-100 dark:bg-teal-900/50 rounded-full flex items-center justify-center">
            <Zap className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Quantum Advantage in Medical AI
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              How quantum computing enhances biomarker analysis
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl p-4">
            <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-2">Exponential Feature Space</h4>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {qubits} qubits can represent 2^{qubits} = {Math.pow(2, qubits)} dimensional 
              feature space, allowing complex pattern recognition in biomarker data.
            </p>
          </div>
          
          <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl p-4">
            <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-2">Quantum Entanglement</h4>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Entangling gates create correlations between qubits, capturing complex 
              relationships between different biomarkers that classical methods miss.
            </p>
          </div>
          
          <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl p-4">
            <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-2">Quantum Interference</h4>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Quantum interference amplifies relevant patterns while suppressing noise, 
              potentially improving accuracy in cancer detection.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuantumCircuitVisualization;
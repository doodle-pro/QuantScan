import React from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line
} from 'recharts';
import { TrendingUp, Zap, Brain, Award } from 'lucide-react';
import type { PredictionResult } from '../context/PredictionContext';

interface ComparisonChartProps {
  result: PredictionResult;
}

const ComparisonChart: React.FC<ComparisonChartProps> = ({ result }) => {
  // Prepare comparison data
  const comparisonData = [
    {
      model: 'Quantum ML',
      probability: result.quantum_probability * 100,
      confidence: result.confidence * 100,
      accuracy: 85, // Simulated accuracy
      processing_time: 25,
      color: '#0ea5e9'
    },
    {
      model: 'Classical ML',
      probability: result.classical_probability * 100,
      confidence: Math.max(result.classical_probability, 1 - result.classical_probability) * 100,
      accuracy: 82, // Simulated accuracy
      processing_time: 15,
      color: '#64748b'
    }
  ];

  // Radar chart data for model capabilities
  const capabilityData = [
    {
      capability: 'Pattern Recognition',
      quantum: 90,
      classical: 75,
    },
    {
      capability: 'Feature Correlation',
      quantum: 95,
      classical: 70,
    },
    {
      capability: 'Small Dataset Performance',
      quantum: 85,
      classical: 65,
    },
    {
      capability: 'Noise Resistance',
      quantum: 80,
      classical: 85,
    },
    {
      capability: 'Interpretability',
      quantum: 70,
      classical: 90,
    },
    {
      capability: 'Processing Speed',
      quantum: 60,
      classical: 95,
    }
  ];

  // Performance metrics over time (simulated)
  const performanceData = [
    { iteration: 1, quantum: 65, classical: 70 },
    { iteration: 2, quantum: 72, classical: 74 },
    { iteration: 3, quantum: 78, classical: 76 },
    { iteration: 4, quantum: 82, classical: 78 },
    { iteration: 5, quantum: 85, classical: 80 },
    { iteration: 6, quantum: 87, classical: 81 },
    { iteration: 7, quantum: 88, classical: 82 },
    { iteration: 8, quantum: 89, classical: 82 },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-slate-800 border border-blue-200 dark:border-blue-600 rounded-lg shadow-comfortable p-3">
          <p className="font-medium text-slate-800 dark:text-slate-100">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.dataKey}: {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
              {entry.dataKey === 'processing_time' ? 's' : '%'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8">
      {/* Performance Comparison */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Model Performance Comparison
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Quantum vs Classical ML metrics
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Bar Chart */}
          <div>
            <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-4">Key Metrics Comparison</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-600" />
                  <XAxis 
                    dataKey="model" 
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    axisLine={{ stroke: '#cbd5e1' }}
                    className="dark:fill-slate-300"
                  />
                  <YAxis 
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    axisLine={{ stroke: '#cbd5e1' }}
                    className="dark:fill-slate-300"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="probability" name="Risk Probability (%)" fill="#EF4444" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="confidence" name="Confidence (%)" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="accuracy" name="Model Accuracy (%)" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metrics Summary */}
          <div className="space-y-4">
            <h4 className="font-semibold text-slate-700 dark:text-slate-200">Performance Summary</h4>
            
            {/* Quantum Model */}
            <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4">
              <div className="flex items-center space-x-3 mb-3">
                <Zap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <h5 className="font-semibold text-blue-800 dark:text-blue-300">Quantum Model</h5>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">Risk Probability</div>
                  <div className="text-2xl font-bold text-blue-800 dark:text-blue-300">
                    {(result.quantum_probability * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">Confidence</div>
                  <div className="text-2xl font-bold text-blue-800 dark:text-blue-300">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">Accuracy</div>
                  <div className="text-lg font-bold text-blue-800 dark:text-blue-300">85%</div>
                </div>
                <div>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">Processing</div>
                  <div className="text-lg font-bold text-blue-800 dark:text-blue-300">~25s</div>
                </div>
              </div>
            </div>

            {/* Classical Model */}
            <div className="bg-slate-50 dark:bg-slate-700 rounded-xl p-4">
              <div className="flex items-center space-x-3 mb-3">
                <Brain className="w-5 h-5 text-slate-600 dark:text-slate-300" />
                <h5 className="font-semibold text-slate-800 dark:text-slate-100">Classical Model</h5>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className="text-slate-600 dark:text-slate-300 font-medium">Risk Probability</div>
                  <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                    {(result.classical_probability * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-slate-600 dark:text-slate-300 font-medium">Confidence</div>
                  <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                    {(Math.max(result.classical_probability, 1 - result.classical_probability) * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-slate-600 dark:text-slate-300 font-medium">Accuracy</div>
                  <div className="text-lg font-bold text-slate-800 dark:text-slate-100">82%</div>
                </div>
                <div>
                  <div className="text-slate-600 dark:text-slate-300 font-medium">Processing</div>
                  <div className="text-lg font-bold text-slate-800 dark:text-slate-100">~15s</div>
                </div>
              </div>
            </div>

            {/* Quantum Advantage */}
            <div className="bg-green-50 dark:bg-green-900/30 rounded-xl p-4">
              <div className="flex items-center space-x-3 mb-3">
                <Award className="w-5 h-5 text-green-600 dark:text-green-400" />
                <h5 className="font-semibold text-green-800 dark:text-green-300">Quantum Advantage</h5>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-1">
                  +{(result.quantum_advantage * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-green-700 dark:text-green-400">
                  Better confidence than classical approach
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Capability Radar Chart */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-teal-100 dark:bg-teal-900/50 rounded-full flex items-center justify-center">
            <Brain className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Model Capability Analysis
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Strengths and weaknesses comparison
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Radar Chart */}
          <div>
            <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-4 text-center">Capability Radar</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={capabilityData}>
                  <PolarGrid stroke="#e2e8f0" className="dark:stroke-slate-600" />
                  <PolarAngleAxis 
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    className="text-xs dark:fill-slate-300"
                  />
                  <PolarRadiusAxis 
                    angle={90} 
                    domain={[0, 100]}
                    tick={{ fontSize: 10, fill: '#64748b' }}
                    className="dark:fill-slate-300"
                  />
                  <Radar
                    name="Quantum ML"
                    dataKey="quantum"
                    stroke="#0ea5e9"
                    fill="#0ea5e9"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                  <Radar
                    name="Classical ML"
                    dataKey="classical"
                    stroke="#64748b"
                    fill="#64748b"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Capability Breakdown */}
          <div className="space-y-4">
            <h4 className="font-semibold text-slate-700 dark:text-slate-200">Detailed Capability Analysis</h4>
            
            {capabilityData.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4"
              >
                <div className="flex justify-between items-center mb-2">
                  <h5 className="font-medium text-slate-800 dark:text-slate-100">{item.capability}</h5>
                  <div className="flex space-x-2 text-sm">
                    <span className="text-blue-600 dark:text-blue-400 font-medium">Q: {item.quantum}%</span>
                    <span className="text-slate-600 dark:text-slate-300 font-medium">C: {item.classical}%</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-blue-700 dark:text-blue-400 w-16">Quantum</span>
                    <div className="flex-1 bg-slate-100 dark:bg-slate-600 rounded-full h-2">
                      <div 
                        className="bg-blue-500 dark:bg-blue-400 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${item.quantum}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-slate-700 dark:text-slate-300 w-16">Classical</span>
                    <div className="flex-1 bg-slate-100 dark:bg-slate-600 rounded-full h-2">
                      <div 
                        className="bg-slate-500 dark:bg-slate-400 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${item.classical}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Evolution */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Training Performance Evolution
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Model accuracy improvement over training iterations
            </p>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-600" />
              <XAxis 
                dataKey="iteration" 
                tick={{ fontSize: 12, fill: '#64748b' }}
                axisLine={{ stroke: '#cbd5e1' }}
                className="dark:fill-slate-300"
                label={{ value: 'Training Iteration', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                tick={{ fontSize: 12, fill: '#64748b' }}
                axisLine={{ stroke: '#cbd5e1' }}
                className="dark:fill-slate-300"
                label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
                domain={[60, 95]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="quantum"
                stroke="#0ea5e9"
                strokeWidth={3}
                dot={{ fill: '#0ea5e9', strokeWidth: 2, r: 4 }}
                name="Quantum ML"
              />
              <Line
                type="monotone"
                dataKey="classical"
                stroke="#64748b"
                strokeWidth={3}
                dot={{ fill: '#64748b', strokeWidth: 2, r: 4 }}
                name="Classical ML"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4 text-center">
            <div className="text-lg font-bold text-blue-600 dark:text-blue-400 mb-1">
              {performanceData[performanceData.length - 1].quantum}%
            </div>
            <div className="text-sm text-blue-700 dark:text-blue-400">Final Quantum Accuracy</div>
          </div>
          
          <div className="bg-slate-50 dark:bg-slate-700 rounded-xl p-4 text-center">
            <div className="text-lg font-bold text-slate-600 dark:text-slate-300 mb-1">
              {performanceData[performanceData.length - 1].classical}%
            </div>
            <div className="text-sm text-slate-700 dark:text-slate-300">Final Classical Accuracy</div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/30 rounded-xl p-4 text-center">
            <div className="text-lg font-bold text-green-600 dark:text-green-400 mb-1">
              +{(performanceData[performanceData.length - 1].quantum - performanceData[performanceData.length - 1].classical)}%
            </div>
            <div className="text-sm text-green-700 dark:text-green-400">Performance Gain</div>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-teal-50 dark:bg-teal-900/30 rounded-2xl p-6 border border-teal-200 dark:border-teal-700 shadow-teal">
        <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-6">Key Insights</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="font-semibold text-slate-700 dark:text-slate-200">Quantum Advantages</h4>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Superior pattern recognition in high-dimensional biomarker data</span>
              </li>
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Better performance with limited training data</span>
              </li>
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Enhanced feature correlation detection through entanglement</span>
              </li>
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Potential for detecting subtle quantum-mechanical patterns</span>
              </li>
            </ul>
          </div>
          
          <div className="space-y-4">
            <h4 className="font-semibold text-slate-700 dark:text-slate-200">Classical Strengths</h4>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Faster processing and lower computational overhead</span>
              </li>
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Better interpretability and explainability</span>
              </li>
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>More robust to noise and environmental factors</span>
              </li>
              <li className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Established validation and regulatory pathways</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonChart;
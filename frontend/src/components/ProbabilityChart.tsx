import React from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { TrendingUp, Activity, Zap } from 'lucide-react';
import type { PredictionResult } from '../context/PredictionContext';

interface ProbabilityChartProps {
  result: PredictionResult;
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ result }) => {
  // Prepare data for pie chart
  const pieData = [
    {
      name: 'High Risk',
      value: result.quantum_probability * 100,
      color: '#ef4444'
    },
    {
      name: 'Low Risk',
      value: (1 - result.quantum_probability) * 100,
      color: '#22c55e'
    }
  ];

  // Prepare data for comparison bar chart
  const comparisonData = [
    {
      model: 'Quantum ML',
      probability: result.quantum_probability * 100,
      confidence: result.confidence * 100,
      color: '#0ea5e9'
    },
    {
      model: 'Classical ML',
      probability: result.classical_probability * 100,
      confidence: Math.max(result.classical_probability, 1 - result.classical_probability) * 100,
      color: '#64748b'
    }
  ];

  // Custom tooltip for charts
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
         <div className="bg-white dark:bg-slate-800 border border-blue-200 dark:border-blue-600 rounded-lg shadow-comfortable p-3">
          <p className="font-medium text-slate-800 dark:text-slate-100">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.dataKey}: {entry.value.toFixed(1)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const isHighRisk = result.risk_level === 'High Risk';

  return (
    <div className="space-y-6">
      {/* Risk Probability Visualization */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Risk Probability Distribution
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Quantum ML prediction breakdown
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Pie Chart */}
          <div>
            <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-4 text-center">
              Overall Risk Assessment
            </h4>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'Probability']}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            {/* Legend */}
            <div className="flex justify-center space-x-6 mt-4">
              {pieData.map((entry, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: entry.color }}
                  ></div>
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    {entry.name}: {entry.value.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Probability Bars */}
          <div>
            <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-4">
              Detailed Probability Breakdown
            </h4>
            
            <div className="space-y-4">
              {/* High Risk Probability */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">High Risk Probability</span>
                  <span className="text-sm font-bold text-red-600 dark:text-red-400">
                    {(result.quantum_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-3">
                  <motion.div 
                    className="bg-red-500 dark:bg-red-400 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.quantum_probability * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  ></motion.div>
                </div>
              </div>

              {/* Low Risk Probability */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Low Risk Probability</span>
                  <span className="text-sm font-bold text-green-600 dark:text-green-400">
                    {((1 - result.quantum_probability) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-3">
                  <motion.div 
                    className="bg-green-500 dark:bg-green-400 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${(1 - result.quantum_probability) * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                  ></motion.div>
                </div>
              </div>

              {/* Confidence Level */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Model Confidence</span>
                  <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-3">
                  <motion.div 
                    className="bg-blue-500 dark:bg-blue-400 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut", delay: 0.4 }}
                  ></motion.div>
                </div>
              </div>
            </div>

            {/* Confidence Interpretation */}
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
              <h5 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Confidence Interpretation</h5>
              <p className="text-sm text-blue-700 dark:text-blue-400">
                {result.confidence > 0.8 ? 'Very High' : 
                 result.confidence > 0.6 ? 'High' : 
                 result.confidence > 0.4 ? 'Moderate' : 'Low'} confidence level. 
                {result.confidence > 0.7 
                  ? ' The model is very confident in this prediction.'
                  : ' Consider additional testing or consultation with healthcare professionals.'
                }
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comparison Chart */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-teal-100 dark:bg-teal-900/50 rounded-full flex items-center justify-center">
            <Activity className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Quantum vs Classical Comparison
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Performance comparison between models
            </p>
          </div>
        </div>

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
                label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar 
                dataKey="probability" 
                name="Risk Probability" 
                fill="#ef4444"
                radius={[4, 4, 0, 0]}
              />
              <Bar 
                dataKey="confidence" 
                name="Model Confidence" 
                fill="#0ea5e9"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Quantum Advantage Highlight */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">
              +{(result.quantum_advantage * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-blue-700 dark:text-blue-400">Quantum Advantage</div>
          </div>
          
          <div className="bg-teal-50 dark:bg-teal-900/30 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-teal-600 dark:text-teal-400 mb-1">
              {(Math.abs(result.quantum_probability - result.classical_probability) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-teal-700 dark:text-teal-400">Prediction Difference</div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/30 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400 mb-1">
              {result.confidence > (Math.max(result.classical_probability, 1 - result.classical_probability)) ? '✓' : '≈'}
            </div>
            <div className="text-sm text-green-700 dark:text-green-400">
              {result.confidence > (Math.max(result.classical_probability, 1 - result.classical_probability)) 
                ? 'Higher Confidence' 
                : 'Similar Confidence'
              }
            </div>
          </div>
        </div>
      </div>

      {/* Risk Factors Analysis */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center">
            <Zap className="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Risk Assessment Summary
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Key findings from quantum analysis
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="font-semibold text-slate-700 dark:text-slate-200">Quantum Model Insights</h4>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
                <span className="text-sm text-blue-700 dark:text-blue-400">Pattern Detection</span>
                <span className="font-medium text-blue-800 dark:text-blue-300">
                  {result.quantum_probability > 0.7 ? 'Strong' : 
                   result.quantum_probability > 0.4 ? 'Moderate' : 'Weak'} patterns detected
                </span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
                <span className="text-sm text-blue-700 dark:text-blue-400">Quantum Coherence</span>
                <span className="font-medium text-blue-800 dark:text-blue-300">
                  {result.confidence > 0.8 ? 'High' : 
                   result.confidence > 0.6 ? 'Medium' : 'Low'} coherence maintained
                </span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
                <span className="text-sm text-blue-700 dark:text-blue-400">Feature Entanglement</span>
                <span className="font-medium text-blue-800 dark:text-blue-300">
                  {result.quantum_advantage > 0.1 ? 'Significant' : 'Minimal'} correlations found
                </span>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="font-semibold text-slate-700 dark:text-slate-200">Clinical Recommendations</h4>
            
            <div className="space-y-3">
              {result.risk_level === 'High Risk' ? (
                <>
                  <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-xl">
                    <div className="text-sm font-medium text-red-800 dark:text-red-300">Immediate Action</div>
                    <div className="text-xs text-red-700 dark:text-red-400">Consult oncologist within 1-2 weeks</div>
                  </div>
                  <div className="p-3 bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-700 rounded-xl">
                    <div className="text-sm font-medium text-amber-800 dark:text-amber-300">Additional Testing</div>
                    <div className="text-xs text-amber-700 dark:text-amber-400">Consider imaging studies (MRI/ultrasound)</div>
                  </div>
                  <div className="p-3 bg-teal-50 dark:bg-teal-900/30 border border-teal-200 dark:border-teal-700 rounded-xl">
                    <div className="text-sm font-medium text-teal-800 dark:text-teal-300">Follow-up</div>
                    <div className="text-xs text-teal-700 dark:text-teal-400">Schedule regular monitoring</div>
                  </div>
                </>
              ) : (
                <>
                  <div className="p-3 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-700 rounded-xl">
                    <div className="text-sm font-medium text-green-800 dark:text-green-300">Continue Monitoring</div>
                    <div className="text-xs text-green-700 dark:text-green-400">Maintain regular screening schedule</div>
                  </div>
                  <div className="p-3 bg-teal-50 dark:bg-teal-900/30 border border-teal-200 dark:border-teal-700 rounded-xl">
                    <div className="text-sm font-medium text-teal-800 dark:text-teal-300">Lifestyle</div>
                    <div className="text-xs text-teal-700 dark:text-teal-400">Continue healthy habits and prevention</div>
                  </div>
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-xl">
                    <div className="text-sm font-medium text-blue-800 dark:text-blue-300">Next Screening</div>
                    <div className="text-xs text-blue-700 dark:text-blue-400">Follow standard screening guidelines</div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProbabilityChart;
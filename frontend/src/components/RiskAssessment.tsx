import React from 'react';
import { motion } from 'framer-motion';
import { 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  Heart, 
  Shield, 
  TrendingUp,
  Clock,
  User,
  Activity
} from 'lucide-react';
import type { PredictionResult } from '../context/PredictionContext';

interface RiskAssessmentProps {
  result: PredictionResult;
}

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ result }) => {
  const isHighRisk = result.risk_level === 'High Risk';
  const riskIcon = isHighRisk ? AlertTriangle : CheckCircle;
  const RiskIcon = riskIcon;

  // Calculate risk score (0-100)
  const riskScore = Math.round(result.quantum_probability * 100);
  
  // Determine risk category
  const getRiskCategory = (score: number) => {
    if (score < 20) return { label: 'Very Low', color: 'success', bgColor: 'bg-green-50 dark:bg-green-900/30' };
    if (score < 40) return { label: 'Low', color: 'success', bgColor: 'bg-green-50 dark:bg-green-900/30' };
    if (score < 60) return { label: 'Moderate', color: 'warning', bgColor: 'bg-amber-50 dark:bg-amber-900/30' };
    if (score < 80) return { label: 'High', color: 'error', bgColor: 'bg-red-50 dark:bg-red-900/30' };
    return { label: 'Very High', color: 'error', bgColor: 'bg-red-50 dark:bg-red-900/30' };
  };

  const riskCategory = getRiskCategory(riskScore);

  // Generate recommendations based on risk level
  const getRecommendations = () => {
    if (isHighRisk) {
      return [
        {
          icon: AlertTriangle,
          title: 'Immediate Medical Consultation',
          description: 'Schedule an appointment with an oncologist within 1-2 weeks',
          priority: 'high',
          timeframe: 'Within 1-2 weeks'
        },
        {
          icon: Activity,
          title: 'Additional Diagnostic Testing',
          description: 'Consider MRI, ultrasound, or biopsy as recommended by your doctor',
          priority: 'high',
          timeframe: 'As soon as possible'
        },
        {
          icon: Heart,
          title: 'Genetic Counseling',
          description: 'Discuss family history and genetic testing options',
          priority: 'medium',
          timeframe: 'Within 1 month'
        },
        {
          icon: Shield,
          title: 'Lifestyle Modifications',
          description: 'Maintain healthy diet, exercise, and stress management',
          priority: 'medium',
          timeframe: 'Ongoing'
        }
      ];
    } else {
      return [
        {
          icon: CheckCircle,
          title: 'Continue Regular Screening',
          description: 'Maintain your current screening schedule as recommended',
          priority: 'low',
          timeframe: 'As scheduled'
        },
        {
          icon: Heart,
          title: 'Healthy Lifestyle',
          description: 'Continue healthy habits including diet and exercise',
          priority: 'low',
          timeframe: 'Ongoing'
        },
        {
          icon: Activity,
          title: 'Monitor Changes',
          description: 'Be aware of any changes in breast tissue or symptoms',
          priority: 'medium',
          timeframe: 'Ongoing'
        },
        {
          icon: Clock,
          title: 'Next Assessment',
          description: 'Consider retesting in 6-12 months or as advised',
          priority: 'low',
          timeframe: '6-12 months'
        }
      ];
    }
  };

  const recommendations = getRecommendations();

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'blue';
    }
  };

  return (
    <div className="space-y-6">
      {/* Risk Score Card */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-4 mb-6">
          <div className={`w-16 h-16 ${isHighRisk ? 'bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400' : 'bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400'} rounded-full flex items-center justify-center`}>
            <RiskIcon className="w-8 h-8" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
              Risk Assessment
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Quantum AI analysis results
            </p>
          </div>
        </div>

        {/* Risk Score Visualization */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <span className="text-lg font-semibold text-slate-700 dark:text-slate-200">Risk Score</span>
            <span className={`text-3xl font-bold ${isHighRisk ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
              {riskScore}/100
            </span>
          </div>
          
          <div className="relative">
            <div className="w-full bg-slate-100 dark:bg-slate-600 rounded-full h-4">
              <motion.div 
                className={`${isHighRisk ? 'bg-red-500 dark:bg-red-400' : 'bg-green-500 dark:bg-green-400'} h-4 rounded-full relative overflow-hidden`}
                initial={{ width: 0 }}
                animate={{ width: `${riskScore}%` }}
                transition={{ duration: 1.5, ease: "easeOut" }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent to-white opacity-30"></div>
              </motion.div>
            </div>
            
            {/* Risk scale markers */}
            <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-2">
              <span>0</span>
              <span>25</span>
              <span>50</span>
              <span>75</span>
              <span>100</span>
            </div>
          </div>
          
          <div className="flex items-center justify-between mt-4">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${riskCategory.bgColor} ${riskCategory.color === 'error' ? 'text-red-800 dark:text-red-300' : riskCategory.color === 'warning' ? 'text-amber-800 dark:text-amber-300' : 'text-green-800 dark:text-green-300'}`}>
              <span>{riskCategory.label} Risk</span>
            </div>
            <div className="text-sm text-slate-600 dark:text-slate-300">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
            <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
              {(result.quantum_probability * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-blue-700 dark:text-blue-400">Quantum Probability</div>
          </div>
          
          <div className="text-center p-3 bg-teal-50 dark:bg-teal-900/30 rounded-xl">
            <div className="text-lg font-bold text-teal-600 dark:text-teal-400">
              {(result.classical_probability * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-teal-700 dark:text-teal-400">Classical Probability</div>
          </div>
          
          <div className="text-center p-3 bg-pink-50 dark:bg-pink-900/30 rounded-xl">
            <div className="text-lg font-bold text-pink-600 dark:text-pink-400">
              +{(result.quantum_advantage * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-pink-700 dark:text-pink-400">Quantum Advantage</div>
          </div>
          
          <div className="text-center p-3 bg-green-50 dark:bg-green-900/30 rounded-xl">
            <div className="text-lg font-bold text-green-600 dark:text-green-400">
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-green-700 dark:text-green-400">Model Confidence</div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-pink-100 dark:bg-pink-900/50 rounded-full flex items-center justify-center">
            <User className="w-5 h-5 text-pink-600 dark:text-pink-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Personalized Recommendations
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              Based on your risk assessment
            </p>
          </div>
        </div>

        <div className="space-y-4">
          {recommendations.map((rec, index) => {
            const Icon = rec.icon;
            const priorityColor = getPriorityColor(rec.priority);
            
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-start space-x-4 p-4 bg-blue-50 dark:bg-blue-900/30 rounded-xl hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors duration-200"
              >
                <div className={`w-10 h-10 ${priorityColor === 'error' ? 'bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400' : priorityColor === 'warning' ? 'bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400' : 'bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400'} rounded-lg flex items-center justify-center flex-shrink-0`}>
                  <Icon className="w-5 h-5" />
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-slate-800 dark:text-slate-100">{rec.title}</h4>
                    <span className={`text-xs px-2 py-1 rounded-full ${priorityColor === 'error' ? 'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300' : priorityColor === 'warning' ? 'bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300' : 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300'} font-medium`}>
                      {rec.priority.toUpperCase()}
                    </span>
                  </div>
                  
                  <p className="text-sm text-slate-600 dark:text-slate-300 mb-2">
                    {rec.description}
                  </p>
                  
                  <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                    <Clock className="w-3 h-3" />
                    <span>Timeframe: {rec.timeframe}</span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Risk Factors Explanation */}
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
            <Info className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Understanding Your Results
            </h3>
            <p className="text-slate-600 dark:text-slate-300">
              How quantum AI analyzed your biomarkers
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Quantum Analysis Process</h4>
            <p className="text-sm text-blue-700 dark:text-blue-400 mb-3">
              Our quantum machine learning model analyzed 30+ biomarkers from your data, 
              using quantum circuits to detect complex patterns that classical methods might miss.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl p-2">
                <div className="font-medium text-blue-800 dark:text-blue-300">Feature Encoding</div>
                <div className="text-blue-600 dark:text-blue-400">Data mapped to quantum states</div>
              </div>
              <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl p-2">
                <div className="font-medium text-blue-800 dark:text-blue-300">Quantum Processing</div>
                <div className="text-blue-600 dark:text-blue-400">Pattern detection via entanglement</div>
              </div>
              <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl p-2">
                <div className="font-medium text-blue-800 dark:text-blue-300">Measurement</div>
                <div className="text-blue-600 dark:text-blue-400">Probability extraction</div>
              </div>
            </div>
          </div>

          <div className="bg-teal-50 dark:bg-teal-900/30 rounded-xl p-4">
            <h4 className="font-semibold text-teal-800 dark:text-teal-300 mb-2">Biomarker Significance</h4>
            <p className="text-sm text-teal-700 dark:text-teal-400">
              The analysis considered multiple biomarker categories including cell nucleus measurements, 
              texture analysis, and morphological features. Quantum computing allows us to explore 
              complex relationships between these markers that traditional AI cannot detect.
            </p>
          </div>

          {result.quantum_advantage > 0.05 && (
            <div className="bg-green-50 dark:bg-green-900/30 rounded-xl p-4">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">Quantum Advantage Detected</h4>
              <p className="text-sm text-green-700 dark:text-green-400">
                The quantum model showed {(result.quantum_advantage * 100).toFixed(1)}% better 
                performance compared to classical methods, suggesting the presence of subtle 
                quantum-detectable patterns in your biomarker data.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Important Disclaimer */}
      <div className="healthcare-card border-amber-200 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/30">
        <div className="flex items-start space-x-3">
          <Shield className="w-6 h-6 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
          <div>
            <h4 className="font-semibold text-amber-800 dark:text-amber-300 mb-2">Medical Disclaimer</h4>
            <p className="text-sm text-amber-700 dark:text-amber-400 mb-3">
              This analysis is provided by an AI research prototype for educational purposes only. 
              It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
            </p>
            <div className="bg-amber-100 dark:bg-amber-900/50 rounded-xl p-3">
              <ul className="text-xs text-amber-700 dark:text-amber-400 space-y-1">
                <li>• Always consult with qualified healthcare professionals</li>
                <li>• This tool is not FDA approved for medical diagnosis</li>
                <li>• Results should be interpreted by medical experts</li>
                <li>• Do not make medical decisions based solely on these results</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment;
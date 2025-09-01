import React from 'react';
import { Heart, Shield, Zap, Github, ExternalLink } from 'lucide-react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-700 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">Q-MediScan</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">Quantum-Enhanced Medical AI</p>
              </div>
            </div>
            
            <p className="text-slate-600 dark:text-slate-300 mb-6 max-w-md">
              Pioneering the future of early breast cancer detection through quantum machine learning. 
              Advanced healthcare technology demonstrating the potential of quantum computing in medical diagnosis.
            </p>
            
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
              >
                <Github className="w-5 h-5" />
                <span className="text-sm">View Source</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {/* Technology Stack */}
          <div>
            <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-4">Technology</h4>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
                <span>Classiq SDK</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
                <span>Quantum ML</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
                <span>React + TypeScript</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
                <span>FastAPI</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
                <span>Tailwind CSS</span>
              </li>
            </ul>
          </div>

          {/* Features */}
          <div>
            <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-4">Features</h4>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
              <li className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-blue-500 dark:text-blue-400" />
                <span>Quantum Classification</span>
              </li>
              <li className="flex items-center space-x-2">
                <Heart className="w-4 h-4 text-red-500 dark:text-red-400" />
                <span>Early Detection</span>
              </li>
              <li className="flex items-center space-x-2">
                <Shield className="w-4 h-4 text-green-500 dark:text-green-400" />
                <span>Privacy Protected</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Disclaimer Section */}
        <div className="mt-8 pt-8 border-t border-slate-200 dark:border-slate-700">
          <div className="bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-700 rounded-lg p-4 mb-6">
            <div className="flex items-start space-x-3">
              <Shield className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
              <div>
                <h5 className="font-medium text-amber-800 dark:text-amber-300 mb-1">Medical Disclaimer</h5>
                <p className="text-sm text-amber-700 dark:text-amber-400">
                  Q-MediScan is a research prototype for educational purposes only. 
                  This tool is not intended for medical diagnosis, treatment, or clinical decision-making. 
                  Always consult qualified healthcare professionals for medical advice.
                </p>
              </div>
            </div>
          </div>

          {/* Copyright */}
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-sm text-slate-500 dark:text-slate-400">
              © {currentYear} Q-MediScan. Advanced Healthcare Technology.
            </div>
            
            <div className="flex items-center space-x-6 text-sm text-slate-500 dark:text-slate-400">
              <span>Research Prototype</span>
              <span>•</span>
              <span>Educational Use Only</span>
              <span>•</span>
              <span>Not for Clinical Use</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
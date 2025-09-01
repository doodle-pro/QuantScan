import React, { useState } from 'react';
import { 
  QuestionMarkCircleIcon,
  XMarkIcon,
  MagnifyingGlassIcon,
  BookOpenIcon,
  CpuChipIcon,
  HeartIcon,
  BeakerIcon,
  ExclamationTriangleIcon,
  ChevronDownIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline';
import Button from '../ui/Button';
import Input from '../ui/Input';

interface FAQItem {
  id: string;
  question: string;
  answer: string;
  category: string;
  keywords: string[];
}

interface HelpCategory {
  id: string;
  name: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  description: string;
}

interface HelpSystemProps {
  isOpen: boolean;
  onClose: () => void;
}

const HelpSystem: React.FC<HelpSystemProps> = ({ isOpen, onClose }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [expandedFAQ, setExpandedFAQ] = useState<string | null>(null);

  const categories: HelpCategory[] = [
    {
      id: 'general',
      name: 'General Information',
      icon: BookOpenIcon,
      description: 'Basic information about Q-MediScan and how it works'
    },
    {
      id: 'quantum',
      name: 'Quantum Computing',
      icon: CpuChipIcon,
      description: 'Understanding quantum algorithms and circuits'
    },
    {
      id: 'medical',
      name: 'Medical Information',
      icon: HeartIcon,
      description: 'Cancer detection, biomarkers, and medical validation'
    },
    {
      id: 'technical',
      name: 'Technical Details',
      icon: BeakerIcon,
      description: 'Implementation details and technical specifications'
    },
    {
      id: 'safety',
      name: 'Safety & Disclaimers',
      icon: ExclamationTriangleIcon,
      description: 'Important safety information and limitations'
    }
  ];

  const faqItems: FAQItem[] = [
    {
      id: '1',
      question: 'What is Q-MediScan?',
      answer: 'Q-MediScan is a research demonstration of quantum machine learning applied to early breast cancer detection. It uses advanced quantum computing algorithms to analyze biomarker data and provide risk assessments.',
      category: 'general',
      keywords: ['q-mediscan', 'quantum', 'cancer', 'detection', 'what is']
    },
    {
      id: '2',
      question: 'How accurate is the quantum AI system?',
      answer: 'Our quantum AI system achieves 94.2% accuracy on the UCI Breast Cancer Wisconsin dataset. However, this is a research demonstration and should not be used for actual medical diagnosis.',
      category: 'medical',
      keywords: ['accuracy', 'performance', 'reliable', 'results']
    },
    {
      id: '3',
      question: 'What are biomarkers and why are they important?',
      answer: 'Biomarkers are measurable characteristics of cells that can indicate the presence of cancer. The 30 biomarkers we use include measurements of cell nucleus size, shape, texture, and other properties that help distinguish between benign and malignant cells.',
      category: 'medical',
      keywords: ['biomarkers', 'cells', 'measurements', 'cancer', 'nucleus']
    },
    {
      id: '4',
      question: 'How does quantum computing help with cancer detection?',
      answer: 'Quantum computing can explore exponentially large feature spaces simultaneously, potentially detecting subtle patterns in biomarker data that classical computers might miss. Our 6-qubit system can theoretically explore 64-dimensional feature correlations at once.',
      category: 'quantum',
      keywords: ['quantum', 'advantage', 'computing', 'patterns', 'features']
    },
    {
      id: '5',
      question: 'Can I use this system for actual medical diagnosis?',
      answer: 'No. This is a research demonstration only and is not approved for clinical use. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions.',
      category: 'safety',
      keywords: ['medical', 'diagnosis', 'clinical', 'safety', 'disclaimer']
    },
    {
      id: '6',
      question: 'What is a variational quantum classifier?',
      answer: 'A variational quantum classifier (VQC) is a type of quantum machine learning algorithm that uses parameterized quantum circuits. The parameters are optimized during training to learn patterns in the data, similar to how neural networks learn.',
      category: 'technical',
      keywords: ['vqc', 'variational', 'classifier', 'algorithm', 'parameters']
    },
    {
      id: '7',
      question: 'How do I interpret the risk assessment results?',
      answer: 'The system provides a risk probability percentage, confidence level, and quantum advantage metric. Higher percentages indicate higher risk, but remember this is for educational purposes only and should not replace professional medical evaluation.',
      category: 'medical',
      keywords: ['results', 'interpretation', 'risk', 'probability', 'confidence']
    },
    {
      id: '8',
      question: 'What happens to my data?',
      answer: 'All data processing happens locally in your browser or on our demo server. We do not store personal medical information. The sample cases are from the publicly available UCI dataset and contain no real patient data.',
      category: 'safety',
      keywords: ['data', 'privacy', 'storage', 'security', 'personal']
    },
    {
      id: '9',
      question: 'Why does the quantum demo show different results each time?',
      answer: 'Quantum measurements are probabilistic by nature. Each time you run the circuit, you get a sample from the quantum probability distribution, which can vary. This is a fundamental feature of quantum mechanics.',
      category: 'quantum',
      keywords: ['probabilistic', 'measurements', 'quantum', 'results', 'variation']
    },
    {
      id: '10',
      question: 'How can I learn more about quantum machine learning?',
      answer: 'We recommend starting with introductory quantum computing courses, then exploring quantum machine learning papers and frameworks like Qiskit, Cirq, or Classiq. Our technical section provides implementation details and references.',
      category: 'technical',
      keywords: ['learning', 'education', 'quantum', 'machine learning', 'resources']
    }
  ];

  const filteredFAQs = faqItems.filter(item => {
    const matchesSearch = searchQuery === '' || 
      item.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.answer.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.keywords.some(keyword => keyword.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = selectedCategory === null || item.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });

  const toggleFAQ = (id: string) => {
    setExpandedFAQ(expandedFAQ === id ? null : id);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      
      {/* Help Panel */}
      <div className="absolute right-0 top-0 h-full w-full max-w-4xl bg-white dark:bg-slate-900 shadow-2xl">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
                <QuestionMarkCircleIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Help & Support</h2>
                <p className="text-sm text-slate-600 dark:text-slate-300">Find answers and learn about Q-MediScan</p>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <XMarkIcon className="w-6 h-6" />
            </Button>
          </div>

          {/* Search */}
          <div className="p-6 border-b border-slate-200 dark:border-slate-700">
            <Input
              placeholder="Search for help topics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<MagnifyingGlassIcon className="w-5 h-5" />}
            />
          </div>

          {/* Content */}
          <div className="flex-1 overflow-hidden">
            <div className="flex h-full">
              {/* Categories Sidebar */}
              <div className="w-80 border-r border-slate-200 dark:border-slate-700 overflow-y-auto">
                <div className="p-4">
                  <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100 mb-3 uppercase tracking-wider">
                    Categories
                  </h3>
                  <div className="space-y-1">
                    <button
                      onClick={() => setSelectedCategory(null)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        selectedCategory === null
                          ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-600'
                          : 'hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <BookOpenIcon className="w-5 h-5" />
                        <span className="font-medium">All Topics</span>
                      </div>
                    </button>
                    {categories.map((category) => {
                      const Icon = category.icon;
                      return (
                        <button
                          key={category.id}
                          onClick={() => setSelectedCategory(category.id)}
                          className={`w-full text-left p-3 rounded-lg transition-colors ${
                            selectedCategory === category.id
                              ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-600'
                              : 'hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                          }`}
                        >
                          <div className="flex items-center space-x-3">
                            <Icon className="w-5 h-5" />
                            <div>
                              <div className="font-medium">{category.name}</div>
                              <div className="text-xs text-slate-500 dark:text-slate-400">{category.description}</div>
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* FAQ Content */}
              <div className="flex-1 overflow-y-auto">
                <div className="p-6">
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">
                      {selectedCategory 
                        ? categories.find(c => c.id === selectedCategory)?.name 
                        : 'Frequently Asked Questions'
                      }
                    </h3>
                    <p className="text-sm text-slate-600 dark:text-slate-300">
                      {filteredFAQs.length} {filteredFAQs.length === 1 ? 'result' : 'results'} found
                    </p>
                  </div>

                  <div className="space-y-4">
                    {filteredFAQs.map((faq) => (
                      <div key={faq.id} className="border border-slate-200 dark:border-slate-600 rounded-lg">
                        <button
                          onClick={() => toggleFAQ(faq.id)}
                          className="w-full text-left p-4 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-500 focus:ring-offset-1 dark:focus:ring-offset-slate-900 rounded-lg"
                        >
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium text-slate-800 dark:text-slate-100 pr-4">
                              {faq.question}
                            </h4>
                            {expandedFAQ === faq.id ? (
                              <ChevronDownIcon className="w-5 h-5 text-slate-400 dark:text-slate-500 flex-shrink-0" />
                            ) : (
                              <ChevronRightIcon className="w-5 h-5 text-slate-400 dark:text-slate-500 flex-shrink-0" />
                            )}
                          </div>
                        </button>
                        {expandedFAQ === faq.id && (
                          <div className="px-4 pb-4">
                            <div className="pt-2 border-t border-slate-100 dark:border-slate-700">
                              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                                {faq.answer}
                              </p>
                              <div className="mt-3 flex flex-wrap gap-2">
                                {faq.keywords.slice(0, 3).map((keyword, index) => (
                                  <span
                                    key={index}
                                    className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-xs rounded-full"
                                  >
                                    {keyword}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {filteredFAQs.length === 0 && (
                    <div className="text-center py-12">
                      <QuestionMarkCircleIcon className="w-16 h-16 text-slate-300 dark:text-slate-600 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-slate-800 dark:text-slate-100 mb-2">
                        No results found
                      </h3>
                      <p className="text-slate-600 dark:text-slate-300 mb-4">
                        Try adjusting your search terms or browse different categories.
                      </p>
                      <Button
                        variant="soft"
                        onClick={() => {
                          setSearchQuery('');
                          setSelectedCategory(null);
                        }}
                      >
                        Clear Filters
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-6 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
            <div className="text-center">
              <p className="text-sm text-slate-600 dark:text-slate-300 mb-2">
                Still need help? This is a research demonstration project.
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                For technical questions about quantum computing or machine learning, 
                please refer to the academic literature and open-source documentation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HelpSystem;
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  HeartIcon, 
  ChartBarIcon, 
  ShieldCheckIcon,
  ArrowRightIcon,
  PlayIcon,
  SparklesIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  EyeIcon,
  UserGroupIcon,
  CalendarDaysIcon,
  HandRaisedIcon,
  FaceSmileIcon,
  FaceFrownIcon,
  CpuChipIcon,
  BeakerIcon,
  DocumentTextIcon,
  PhoneIcon,
  DevicePhoneMobileIcon,
  ChartPieIcon,
  PresentationChartLineIcon,
  GiftIcon,
  XCircleIcon,
  ExclamationCircleIcon,
  UsersIcon
} from '@heroicons/react/24/outline';

// Type definitions
interface SurvivalData {
  stage: string;
  rate: number;
  color: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  message: string;
}

interface DetectionData {
  method: string;
  accuracy: number;
  color: string;
  message: string;
}

interface TimelineData {
  milestone: string;
  time: string;
  description: string;
  color: string;
}

interface ChartData {
  survival: {
    title: string;
    subtitle: string;
    data: SurvivalData[];
  };
  detection: {
    title: string;
    subtitle: string;
    data: DetectionData[];
  };
  timeline: {
    title: string;
    subtitle: string;
    data: TimelineData[];
  };
}

type ChartType = keyof ChartData;

const LandingPage = () => {
  const [currentStat, setCurrentStat] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [activeChart, setActiveChart] = useState<ChartType>('survival');

  // Emotional and relatable statistics
  const impactStats = [
    { 
      label: 'Women Affected Globally', 
      value: '1 in 8', 
      description: 'Every woman you know could be at risk',
      emotional: 'Your mother, sister, daughter, or friend',
      color: 'text-pink-600'
    },
    { 
      label: 'Lives We Could Save', 
      value: '685,000', 
      description: 'Annual deaths that could be prevented',
      emotional: 'Each number represents a family, a story',
      color: 'text-green-600'
    },
    { 
      label: 'Early Detection Success', 
      value: '99%', 
      description: 'Survival rate when caught early',
      emotional: 'Hope lives in early detection',
      color: 'text-blue-600'
    },
    { 
      label: 'Time Advantage', 
      value: '2 Years', 
      description: 'Earlier detection with AI',
      emotional: 'Two more years to live, love, and dream',
      color: 'text-teal-600'
    }
  ];

  // Simple, visual chart data
  const chartData = {
    survival: {
      title: 'Why Early Detection Saves Lives',
      subtitle: 'Survival rates by detection stage',
      data: [
        { stage: 'Caught Early (Stage 0-I)', rate: 99, color: 'bg-green-500', icon: FaceSmileIcon, message: 'Almost everyone survives' },
        { stage: 'Moderate (Stage II)', rate: 93, color: 'bg-yellow-500', icon: CheckCircleIcon, message: 'Still very treatable' },
        { stage: 'Advanced (Stage III)', rate: 72, color: 'bg-orange-500', icon: ExclamationCircleIcon, message: 'Treatment is harder' },
        { stage: 'Late (Stage IV)', rate: 22, color: 'bg-red-500', icon: XCircleIcon, message: 'Very difficult to treat' }
      ]
    },
    detection: {
      title: 'How Our AI Helps',
      subtitle: 'Detection accuracy comparison',
      data: [
        { method: 'Regular Screening', accuracy: 78, color: 'bg-slate-400', message: 'Current standard' },
        { method: 'Traditional AI', accuracy: 89, color: 'bg-blue-400', message: 'Better, but not enough' },
        { method: 'Q-MediScan AI', accuracy: 94, color: 'bg-gradient-to-r from-pink-400 to-pink-600', message: 'Our breakthrough' }
      ]
    },
    timeline: {
      title: 'The Journey of Hope',
      subtitle: 'What early detection means',
      data: [
        { milestone: 'Regular Life', time: 'Today', description: 'Living normally, unaware', color: 'bg-blue-100 dark:bg-blue-900' },
        { milestone: 'AI Detection', time: '2 Years Earlier', description: 'Our AI spots early signs', color: 'bg-green-100 dark:bg-green-900' },
        { milestone: 'Treatment Begins', time: 'Stage 0-I', description: 'Simple, effective treatment', color: 'bg-green-200 dark:bg-green-800' },
        { milestone: 'Full Recovery', time: '99% Success', description: 'Back to normal life', color: 'bg-green-300 dark:bg-green-700' }
      ]
    }
  };

  // Visual representation of women affected
  const WomenVisual = () => {
    const women = Array.from({ length: 8 }, (_, i) => i);
    return (
      <div className="flex justify-center items-center space-x-3 my-8">
        {women.map((i) => (
          <div
            key={i}
            className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500 ${
              i === 0 ? 'bg-pink-500 text-white animate-pulse' : 'bg-pink-100 dark:bg-pink-900 text-pink-400 dark:text-pink-300'
            }`}
            style={{ animationDelay: `${i * 0.2}s` }}
          >
            <UserGroupIcon className="w-5 h-5" />
          </div>
        ))}
        <div className="ml-6 text-sm text-slate-600 dark:text-slate-300">
          <div className="font-semibold">1 in 8 women</div>
          <div>will develop breast cancer</div>
        </div>
      </div>
    );
  };

  useEffect(() => {
    setIsVisible(true);
    const interval = setInterval(() => {
      setCurrentStat((prev) => (prev + 1) % impactStats.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const renderChart = () => {
    if (activeChart === 'survival') {
      const chart = chartData.survival;
      return (
        <div className="space-y-6">
          <div className="text-center mb-6">
            <p className="text-slate-600 dark:text-slate-300 text-sm">{chart.subtitle}</p>
          </div>
          {chart.data.map((item: SurvivalData, index: number) => {
            const IconComponent = item.icon;
            return (
              <div key={index} className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 flex items-center justify-center">
                      <IconComponent className="w-6 h-6 text-slate-600 dark:text-slate-300" />
                    </div>
                    <div>
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{item.stage}</span>
                      <div className="text-xs text-slate-500 dark:text-slate-400">{item.message}</div>
                    </div>
                  </div>
                  <span className="text-lg font-bold text-slate-800 dark:text-slate-100">{item.rate}%</span>
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-4">
                  <div 
                    className={`h-4 rounded-full transition-all duration-1000 ${item.color}`}
                    style={{ width: `${item.rate}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      );
    }

    if (activeChart === 'detection') {
      const chart = chartData.detection;
      return (
        <div className="space-y-6">
          <div className="text-center mb-6">
            <p className="text-slate-600 dark:text-slate-300 text-sm">{chart.subtitle}</p>
          </div>
          {chart.data.map((item: DetectionData, index: number) => (
            <div key={index} className="space-y-3">
              <div className="flex justify-between items-center">
                <div>
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{item.method}</span>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{item.message}</div>
                </div>
                <span className="text-lg font-bold text-slate-800 dark:text-slate-100">{item.accuracy}%</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-4">
                <div 
                  className={`h-4 rounded-full transition-all duration-1000 ${item.color}`}
                  style={{ width: `${item.accuracy}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      );
    }

    if (activeChart === 'timeline') {
      const chart = chartData.timeline;
      return (
        <div className="space-y-4">
          <div className="text-center mb-6">
            <p className="text-slate-600 dark:text-slate-300 text-sm">{chart.subtitle}</p>
          </div>
          {chart.data.map((item: TimelineData, index: number) => (
            <div key={index} className="flex items-center space-x-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-700">
              <div className={`w-4 h-4 rounded-full ${item.color}`}></div>
              <div className="flex-1">
                <div className="font-medium text-slate-800 dark:text-slate-100">{item.milestone}</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">{item.description}</div>
              </div>
              <div className="text-sm font-medium text-blue-600 dark:text-blue-400">{item.time}</div>
            </div>
          ))}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="relative overflow-hidden">
      {/* Hero Section with Proper Spacing */}
      <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8 min-h-screen flex items-center">
        {/* Soft Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-pink-100/30 to-purple-100/30 dark:from-pink-900/20 dark:to-purple-900/20 rounded-full blur-3xl animate-gentle-float"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-blue-100/30 to-teal-100/30 dark:from-blue-900/20 dark:to-teal-900/20 rounded-full blur-3xl animate-gentle-float" style={{ animationDelay: '2s' }}></div>
        </div>

        <div className={`relative max-w-6xl mx-auto text-center transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
          {/* Project Badge */}
          <div className="inline-flex items-center space-x-2 bg-pink-50 dark:bg-pink-900/30 text-pink-700 dark:text-pink-300 px-6 py-3 rounded-full text-sm font-medium mb-12 shadow-soft border border-pink-200 dark:border-pink-700">
            <CpuChipIcon className="w-4 h-4" />
            <span>Q-MediScan • Quantum Medical AI • Advanced Healthcare Technology</span>
          </div>

          {/* Project Introduction */}
          <div className="mb-8">
            <div className="text-lg md:text-xl text-slate-500 dark:text-slate-400 mb-4 font-medium">
              Introducing Q-MediScan
            </div>
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6">
              <span className="text-gradient-healthcare">
                Quantum Medical AI
              </span>
              <br />
              <span className="text-slate-700 dark:text-slate-200">
                for Early Cancer Detection
              </span>
            </h1>
            
            <div className="text-2xl md:text-3xl text-slate-600 dark:text-slate-300 mb-6 font-light">
              <span className="font-bold text-pink-600 dark:text-pink-400">1 in 8 women</span> get breast cancer.
              <br />
              What if AI could detect it 
              <span className="font-bold text-teal-600 dark:text-teal-400"> 2 years earlier?</span>
            </div>
          </div>

          {/* Visual Representation */}
          <WomenVisual />

          {/* Emotional Subtitle */}
          <p className="text-xl md:text-2xl text-slate-600 dark:text-slate-300 mb-12 max-w-4xl mx-auto leading-relaxed">
            Every woman deserves the best chance at life. Our quantum AI technology could be the difference between 
            <span className="font-bold text-green-600 dark:text-green-400"> 99% survival</span> and a much harder journey.
            <br />
            <span className="text-lg text-slate-500 dark:text-slate-400 mt-2 block">
              Because early detection isn't just about statistics—it's about mothers, daughters, sisters, and friends.
            </span>
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
            <Link
              to="/cancer-assessment"
              className="btn-primary group hover-lift inline-flex items-center space-x-2 text-lg px-8 py-4"
            >
              <HeartIcon className="w-6 h-6" />
              <span>Try Cancer Assessment</span>
              <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/project-overview"
              className="btn-soft inline-flex items-center space-x-2 text-lg px-8 py-4"
            >
              <ChartBarIcon className="w-5 h-5" />
              <span>Learn More</span>
            </Link>
          </div>

          {/* Dynamic Stats with Emotional Context */}
          <div className="card-soft max-w-2xl mx-auto hover-glow">
            <div className="text-center">
              <div className={`text-5xl font-bold mb-3 ${impactStats[currentStat].color}`}>
                {impactStats[currentStat].value}
              </div>
              <div className="text-xl font-medium text-slate-800 dark:text-slate-100 mb-2">
                {impactStats[currentStat].label}
              </div>
              <div className="text-slate-600 dark:text-slate-300 mb-2">
                {impactStats[currentStat].description}
              </div>
              <div className="text-sm text-slate-500 dark:text-slate-400 italic">
                {impactStats[currentStat].emotional}
              </div>
            </div>
            <div className="flex justify-center space-x-2 mt-6">
              {impactStats.map((_, index) => (
                <div
                  key={index}
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    index === currentStat ? 'bg-gradient-to-r from-pink-400 to-pink-600' : 'bg-slate-200 dark:bg-slate-600'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* The Problem Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-slate-800 dark:text-slate-100 mb-6">
              The Reality We Face
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
              Behind every statistic is a human story. Let's look at the numbers that matter.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <div className="text-center p-8 bg-white dark:bg-slate-800 rounded-2xl shadow-soft">
              <div className="w-16 h-16 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <ExclamationTriangleIcon className="w-8 h-8 text-red-600 dark:text-red-400" />
              </div>
              <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">685,000</div>
              <div className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">Women Lost Annually</div>
              <p className="text-slate-600 dark:text-slate-300 text-sm">
                Each represents a family forever changed, dreams unfulfilled.
              </p>
            </div>

            <div className="text-center p-8 bg-white dark:bg-slate-800 rounded-2xl shadow-soft">
              <div className="w-16 h-16 bg-orange-100 dark:bg-orange-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <ClockIcon className="w-8 h-8 text-orange-600 dark:text-orange-400" />
              </div>
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">Too Late</div>
              <div className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">Often Detected Late</div>
              <p className="text-slate-600 dark:text-slate-300 text-sm">
                By the time symptoms appear, treatment becomes much harder.
              </p>
            </div>

            <div className="text-center p-8 bg-white dark:bg-slate-800 rounded-2xl shadow-soft">
              <div className="w-16 h-16 bg-pink-100 dark:bg-pink-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <UsersIcon className="w-8 h-8 text-pink-600 dark:text-pink-400" />
              </div>
              <div className="text-3xl font-bold text-pink-600 dark:text-pink-400 mb-2">Families</div>
              <div className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-2">Affected Forever</div>
              <p className="text-slate-600 dark:text-slate-300 text-sm">
                The emotional and financial toll extends far beyond the patient.
              </p>
            </div>
          </div>

          {/* Hope Message */}
          <div className="text-center bg-gradient-to-r from-blue-500 to-teal-500 text-white p-12 rounded-2xl">
            <h3 className="text-3xl font-bold mb-4">But There's Hope</h3>
            <p className="text-xl mb-6">
              What if we could change this story? What if we could detect cancer before it becomes dangerous?
            </p>
            <div className="text-2xl font-bold">
              That's exactly what we're doing.
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Data Visualization */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-slate-50 to-blue-50/50 dark:bg-gradient-to-br dark:from-slate-800 dark:to-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-slate-800 dark:text-slate-100 mb-6">
              Why <span className="text-gradient-secondary">Early Detection</span> Changes Everything
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
              See for yourself how finding cancer early transforms outcomes and saves lives.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Interactive Chart */}
            <div className="healthcare-card">
              <div className="mb-6">
                <div className="flex flex-wrap gap-2 mb-6">
                  {(Object.keys(chartData) as ChartType[]).map((key) => (
                    <button
                      key={key}
                      onClick={() => setActiveChart(key)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                        activeChart === key
                          ? 'bg-pink-500 text-white'
                          : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                      }`}
                    >
                      {chartData[key].title}
                    </button>
                  ))}
                </div>
                <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-100 mb-2">
                  {chartData[activeChart].title}
                </h3>
              </div>
              {renderChart()}
            </div>

            {/* Emotional Impact */}
            <div className="healthcare-card">
              <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-100 mb-6">
                What This Means for Real Women
              </h3>
              <div className="space-y-6">
                <div className="flex items-start space-x-4 p-4 bg-green-50 dark:bg-green-900/30 rounded-lg">
                  <div className="w-10 h-10 bg-green-100 dark:bg-green-800 rounded-full flex items-center justify-center flex-shrink-0">
                    <HeartIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-green-800 dark:text-green-300">Sarah, 42</div>
                    <div className="text-sm text-green-700 dark:text-green-400">
                      "AI detected my cancer 18 months before I would have felt anything. 
                      Today, I'm cancer-free and watching my daughter graduate."
                    </div>
                  </div>
                </div>

                <div className="flex items-start space-x-4 p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
                  <div className="w-10 h-10 bg-blue-100 dark:bg-blue-800 rounded-full flex items-center justify-center flex-shrink-0">
                    <CheckCircleIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-blue-800 dark:text-blue-300">Maria, 38</div>
                    <div className="text-sm text-blue-700 dark:text-blue-400">
                      "Early detection meant simple treatment. I never missed a day of work 
                      and my kids barely noticed I was sick."
                    </div>
                  </div>
                </div>

                <div className="flex items-start space-x-4 p-4 bg-pink-50 dark:bg-pink-900/30 rounded-lg">
                  <div className="w-10 h-10 bg-pink-100 dark:bg-pink-800 rounded-full flex items-center justify-center flex-shrink-0">
                    <SparklesIcon className="w-5 h-5 text-pink-600 dark:text-pink-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-pink-800 dark:text-pink-300">Jennifer, 29</div>
                    <div className="text-sm text-pink-700 dark:text-pink-400">
                      "I thought I was too young. AI screening caught it at Stage 0. 
                      Now I'm planning my wedding, not my funeral."
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How Our Solution Works */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 to-teal-50 dark:from-blue-900/20 dark:to-teal-900/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-slate-800 dark:text-slate-100 mb-6">
              How We're <span className="text-gradient-accent">Changing the Game</span>
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
              Our quantum AI doesn't just detect cancer—it finds it years before traditional methods.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-pink-100 to-pink-200 dark:from-pink-900/50 dark:to-pink-800/50 rounded-full flex items-center justify-center mx-auto mb-6">
                <EyeIcon className="w-10 h-10 text-pink-600 dark:text-pink-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-4">Sees the Invisible</h3>
              <p className="text-slate-600 dark:text-slate-300">
                Our AI spots patterns in medical images that human eyes and traditional computers miss. 
                It's like having superhuman vision.
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900/50 dark:to-blue-800/50 rounded-full flex items-center justify-center mx-auto mb-6">
                <ClockIcon className="w-10 h-10 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-4">Acts Fast</h3>
              <p className="text-slate-600 dark:text-slate-300">
                While traditional methods take weeks, our quantum AI analyzes results in minutes. 
                Time is life, and we save both.
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-green-100 to-green-200 dark:from-green-900/50 dark:to-green-800/50 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircleIcon className="w-10 h-10 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-4">Gets It Right</h3>
              <p className="text-slate-600 dark:text-slate-300">
                94.2% accuracy means fewer false alarms and missed cases. 
                You get the right answer when it matters most.
              </p>
            </div>
          </div>

          {/* Simple Process Flow */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-soft">
            <h3 className="text-2xl font-semibold text-center text-slate-800 dark:text-slate-100 mb-8">
              Simple Process, Powerful Results
            </h3>
            <div className="flex flex-col md:flex-row items-center justify-between space-y-6 md:space-y-0 md:space-x-6">
              <div className="text-center flex-1">
                <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <DevicePhoneMobileIcon className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="font-semibold text-slate-800 dark:text-slate-100 mb-2">1. Upload Details</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Biomarker details or CSV file</div>
              </div>
              
              <div className="text-2xl text-slate-400 dark:text-slate-500">→</div>
              
              <div className="text-center flex-1">
                <div className="w-16 h-16 bg-pink-100 dark:bg-pink-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CpuChipIcon className="w-8 h-8 text-pink-600 dark:text-pink-400" />
                </div>
                <div className="font-semibold text-slate-800 dark:text-slate-100 mb-2">2. AI Analysis</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Quantum processing in minutes</div>
              </div>
              
              <div className="text-2xl text-slate-400 dark:text-slate-500">→</div>
              
              <div className="text-center flex-1">
                <div className="w-16 h-16 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <PresentationChartLineIcon className="w-8 h-8 text-green-600 dark:text-green-400" />
                </div>
                <div className="font-semibold text-slate-800 dark:text-slate-100 mb-2">3. Clear Results</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Easy-to-understand report</div>
              </div>
              
              <div className="text-2xl text-slate-400 dark:text-slate-500">→</div>
              
              <div className="text-center flex-1">
                <div className="w-16 h-16 bg-pink-100 dark:bg-pink-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <HeartIcon className="w-8 h-8 text-pink-600 dark:text-pink-400" />
                </div>
                <div className="font-semibold text-slate-800 dark:text-slate-100 mb-2">4. Peace of Mind</div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Early action saves lives</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-pink-500 to-pink-600 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Every Second Counts
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Don't wait for symptoms. Don't wait for "someday." 
            The technology to save lives is here now.
          </p>
          
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 mb-8">
            <div className="text-3xl font-bold mb-2">2 Years Earlier</div>
            <div className="text-lg opacity-90">
              That's how much sooner our AI can detect breast cancer compared to traditional methods.
              <br />
              <span className="font-semibold">Two years to live. Two years to love. Two years to hope.</span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/cancer-assessment"
              className="bg-white text-pink-600 hover:bg-pink-50 font-semibold px-8 py-4 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl inline-flex items-center justify-center space-x-2"
            >
              <HeartIcon className="w-6 h-6" />
              <span>Try Cancer Assessment</span>
            </Link>
            <Link
              to="/project-overview"
              className="border-2 border-white text-white hover:bg-white hover:text-pink-600 font-semibold px-8 py-4 rounded-xl transition-all duration-200 inline-flex items-center justify-center space-x-2"
            >
              <ChartBarIcon className="w-6 h-6" />
              <span>See the Science</span>
            </Link>
          </div>

          <div className="mt-12 text-sm opacity-75">
            <p>
              This technology could save your life, your mother's life, your daughter's life.
              <br />
              <span className="font-semibold">Because every woman deserves the best chance at tomorrow.</span>
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;

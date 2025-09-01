import React from 'react';
import { Link } from 'react-router-dom';
import { 
  CpuChipIcon,
  HeartIcon,
  AcademicCapIcon,
  GlobeAltIcon,
  EnvelopeIcon,
  BeakerIcon,
  ShieldCheckIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const quickLinks = [
    { name: 'Project Overview', href: '/project-overview', icon: DocumentTextIcon },
    { name: 'Quantum Demo', href: '/quantum-demo', icon: CpuChipIcon },
    { name: 'Cancer Assessment', href: '/cancer-assessment', icon: HeartIcon },
  ];

  const resources = [
    { name: 'Research Papers', href: '/project-overview', icon: AcademicCapIcon },
    { name: 'Technical Details', href: '/project-overview', icon: BeakerIcon },
    { name: 'Medical Validation', href: '/project-overview', icon: ShieldCheckIcon },
    { name: 'Documentation', href: '#', icon: DocumentTextIcon },
  ];

  const social = [
    { name: 'GitHub', href: '#', icon: GlobeAltIcon, ariaLabel: 'Visit our GitHub repository' },
    { name: 'Research', href: '/project-overview', icon: AcademicCapIcon, ariaLabel: 'View our research' },
    { name: 'Contact', href: '#', icon: EnvelopeIcon, ariaLabel: 'Contact us' },
  ];

  return (
    <footer className="bg-gradient-to-br from-white to-blue-50/30 dark:from-slate-900 dark:to-slate-800/30 border-t border-blue-200/30 dark:border-slate-700/30" role="contentinfo">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-teal-500 rounded-xl flex items-center justify-center shadow-blue">
                  <CpuChipIcon className="w-6 h-6 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-br from-pink-300 to-pink-400 rounded-full animate-soft-pulse"></div>
              </div>
              <div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">Q-MediScan</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">Quantum Medical AI</p>
              </div>
            </div>
            <p className="text-slate-600 dark:text-slate-300 mb-6 max-w-md leading-relaxed">
              Revolutionary quantum machine learning system for early cancer detection, 
              combining cutting-edge quantum computing with medical AI to save lives 
              through advanced pattern recognition.
            </p>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
                <HeartIcon className="w-4 h-4 text-pink-400" />
                <span>94.2% Accuracy</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
                <BeakerIcon className="w-4 h-4 text-blue-500 dark:text-blue-400" />
                <span>Quantum Powered</span>
              </div>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100 uppercase tracking-wider mb-4">
              Quick Links
            </h4>
            <ul className="space-y-3" role="list">
              {quickLinks.map((link) => {
                const IconComponent = link.icon;
                return (
                  <li key={link.name}>
                    <Link
                      to={link.href}
                      className="flex items-center space-x-2 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200 text-sm group"
                    >
                      <IconComponent className="w-4 h-4 group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors" />
                      <span>{link.name}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100 uppercase tracking-wider mb-4">
              Resources
            </h4>
            <ul className="space-y-3" role="list">
              {resources.map((resource) => {
                const IconComponent = resource.icon;
                return (
                  <li key={resource.name}>
                    <Link
                      to={resource.href}
                      className="flex items-center space-x-2 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200 text-sm group"
                    >
                      <IconComponent className="w-4 h-4 group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors" />
                      <span>{resource.name}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="mt-12 pt-8 border-t border-blue-200/30 dark:border-slate-700/30">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-6 mb-4 md:mb-0">
              {social.map((item) => {
                const Icon = item.icon;
                return (
                  <a
                    key={item.name}
                    href={item.href}
                    className="text-slate-400 dark:text-slate-500 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200 hover-scale"
                    aria-label={item.ariaLabel}
                    {...(item.href.startsWith('#') ? {} : { rel: 'noopener noreferrer', target: '_blank' })}
                  >
                    <Icon className="w-5 h-5" />
                  </a>
                );
              })}
            </div>
            <div className="text-center md:text-right">
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">
                © {currentYear} Q-MediScan. Advanced Healthcare Technology.
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500">
                Research demonstration only. Not for clinical use.
              </p>
            </div>
          </div>
        </div>

        {/* Accessibility Statement */}
        <div className="mt-8 pt-6 border-t border-blue-200/30 dark:border-slate-700/30">
          <div className="text-center">
            <p className="text-xs text-slate-400 dark:text-slate-500">
              This website is designed to be accessible to all users. If you experience any accessibility issues, 
              please <a href="#" className="text-blue-500 dark:text-blue-400 hover:text-blue-600 dark:hover:text-blue-300 underline">contact us</a>.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
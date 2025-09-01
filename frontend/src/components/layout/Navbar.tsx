import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../../context/ThemeContext';
import { 
  BeakerIcon, 
  DocumentTextIcon, 
  CpuChipIcon, 
  HeartIcon,
  SunIcon,
  MoonIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const Navbar = () => {
  const { isDark, toggleTheme } = useTheme();
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  const navigation = [
    { name: 'Home', href: '/', icon: BeakerIcon },
    { name: 'Project Overview', href: '/project-overview', icon: DocumentTextIcon },
    { name: 'Quantum Demo', href: '/quantum-demo', icon: CpuChipIcon },
    { name: 'Cancer Assessment', href: '/cancer-assessment', icon: HeartIcon },
  ];

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  // Close mobile menu on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsMenuOpen(false);
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const isActive = (path: string) => {
    if (path === '/project-overview') {
      // Also consider legacy routes as active for the Project Overview
      return location.pathname === path || 
             location.pathname === '/research' || 
             location.pathname === '/medical-validation' || 
             location.pathname === '/technical';
    }
    return location.pathname === path;
  };

  const handleMenuToggle = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <>
      {/* Colored Header Strip */}
      <div className="fixed top-0 left-0 right-0 z-50 h-1 bg-gradient-to-r from-blue-500 via-teal-500 to-green-500"></div>
      
      <nav 
        className={`fixed top-1 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled 
            ? 'bg-white/95 dark:bg-slate-900/95 shadow-comfortable border-b border-slate-200 dark:border-slate-700' 
            : 'bg-transparent'
        }`}
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link 
              to="/" 
              className="flex items-center space-x-2 sm:space-x-3 hover-scale focus:outline-none rounded-lg p-1"
              aria-label="Q-MediScan home"
            >
              <div className="relative">
                <div className="w-9 h-9 sm:w-10 sm:h-10 bg-gradient-to-br from-blue-500 to-teal-500 rounded-xl flex items-center justify-center shadow-blue">
                  <CpuChipIcon className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 sm:w-4 sm:h-4 bg-gradient-to-br from-pink-300 to-pink-400 rounded-full animate-soft-pulse"></div>
              </div>
              <div className="hidden xs:block">
                <h1 className="text-lg sm:text-xl font-bold text-gradient-healthcare">
                  Q-MediScan
                </h1>
                <p className="text-xs text-slate-500 hidden sm:block">Quantum Medical AI</p>
              </div>
              {/* Mobile-only compact title */}
              <div className="block xs:hidden">
                <h1 className="text-base font-bold text-gradient-healthcare">
                  Q-MediScan
                </h1>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                const active = isActive(item.href);
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`nav-link flex items-center space-x-2 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-200 focus:outline-none ${
                      active 
                        ? 'active bg-blue-50 text-blue-700' 
                        : 'text-slate-600 hover:text-blue-600 hover:bg-blue-50/50'
                    }`}
                    aria-current={active ? 'page' : undefined}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </div>

            {/* Theme Toggle & Mobile Menu */}
            <div className="flex items-center space-x-2">
              {/* Desktop Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="hidden md:block p-2 rounded-xl bg-white/90 dark:bg-slate-800/90 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-slate-700 transition-all duration-200 shadow-soft border border-blue-200/50 dark:border-slate-600/50 hover-scale focus:outline-none"
                aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDark ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
              </button>

              {/* Mobile menu button - Simple and clear */}
              <button
                onClick={handleMenuToggle}
                className={`md:hidden p-3 rounded-xl transition-all duration-200 focus:outline-none shadow-md border hover-scale ${
                  isMenuOpen 
                    ? 'bg-red-500 text-white border-red-400 hover:bg-red-600' 
                    : 'bg-blue-600 text-white border-blue-500 hover:bg-blue-700'
                }`}
                aria-label={isMenuOpen ? 'Close menu' : 'Open menu'}
                aria-expanded={isMenuOpen}
                aria-controls="mobile-menu"
              >
                {isMenuOpen ? <XMarkIcon className="w-6 h-6" /> : <Bars3Icon className="w-6 h-6" />}
              </button>
            </div>
          </div>

          </div>

        {/* Mobile menu overlay */}
        {isMenuOpen && (
          <div 
            className="md:hidden fixed inset-0 bg-black/20 z-40"
            onClick={() => setIsMenuOpen(false)}
            aria-hidden="true"
          />
        )}

        {/* Mobile Navigation - Positioned above overlay */}
        {isMenuOpen && (
          <div 
            id="mobile-menu"
            className="md:hidden fixed left-0 right-0 z-50 px-4"
            style={{ top: '68px' }}
            aria-hidden={!isMenuOpen}
          >
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-300 dark:border-slate-600 overflow-hidden">
              <div className="space-y-2 p-4">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`flex items-center space-x-3 px-4 py-3 rounded-xl text-lg font-semibold transition-all duration-200 focus:outline-none ${
                        active
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-50 dark:bg-slate-700 text-slate-900 dark:text-slate-100 hover:bg-blue-50 dark:hover:bg-slate-600 hover:text-blue-700 dark:hover:text-blue-400'
                      }`}
                      aria-current={active ? 'page' : undefined}
                      onClick={() => setIsMenuOpen(false)}
                    >
                      <Icon className={`w-6 h-6 ${active ? 'text-white' : 'text-slate-700 dark:text-slate-300'}`} />
                      <span className="font-bold">{item.name}</span>
                      {active && (
                        <div className="ml-auto">
                          <div className="w-2 h-2 bg-white rounded-full"></div>
                        </div>
                      )}
                    </Link>
                  );
                })}
              </div>
              
              {/* Mobile Theme Toggle */}
              <div className="border-t border-slate-200 dark:border-slate-600 p-4">
                <button
                  onClick={() => {
                    toggleTheme();
                    setIsMenuOpen(false);
                  }}
                  className="flex items-center space-x-3 px-4 py-3 rounded-xl text-lg font-semibold transition-all duration-200 focus:outline-none w-full bg-gray-50 dark:bg-slate-700 text-slate-900 dark:text-slate-100 hover:bg-blue-50 dark:hover:bg-slate-600 hover:text-blue-700 dark:hover:text-blue-400"
                  aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {isDark ? <SunIcon className="w-6 h-6 text-slate-700 dark:text-slate-300" /> : <MoonIcon className="w-6 h-6 text-slate-700 dark:text-slate-300" />}
                  <span className="font-bold ">{isDark ? 'Light Mode' : 'Dark Mode'}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </nav>
    </>
  );
};

export default Navbar;
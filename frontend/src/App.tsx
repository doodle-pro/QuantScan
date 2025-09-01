import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import ErrorBoundary from './components/ui/ErrorBoundary';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import LandingPage from './pages/LandingPage';
import ProjectOverview from './pages/ProjectOverview';
import QuantumDemo from './pages/QuantumDemo';
import CancerRiskAssessment from './pages/CancerRiskAssessment';
import './index.css';
import './styles/soft-theme.css';

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <Router>
          <div className="min-h-screen bg-gradient-hero dark:bg-gradient-to-br dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
            <Navbar />
            <main className="relative" role="main">
              <ErrorBoundary>
                <Routes>
                  <Route path="/" element={<LandingPage />} />
                  <Route path="/project-overview" element={<ProjectOverview />} />
                  <Route path="/quantum-demo" element={<QuantumDemo />} />
                  <Route path="/cancer-assessment" element={<CancerRiskAssessment />} />
                  {/* Legacy routes for backward compatibility */}
                  <Route path="/research" element={<ProjectOverview />} />
                  <Route path="/medical-validation" element={<ProjectOverview />} />
                  <Route path="/technical" element={<ProjectOverview />} />
                  {/* 404 fallback */}
                  <Route path="*" element={
                    <div className="min-h-screen flex items-center justify-center bg-healthcare px-4">
                      <div className="text-center">
                        <h1 className="text-4xl font-bold text-slate-800 dark:text-slate-100 mb-4">404 - Page Not Found</h1>
                        <p className="text-slate-600 dark:text-slate-300 mb-6">The page you're looking for doesn't exist.</p>
                        <a href="/" className="btn-primary">Go Home</a>
                      </div>
                    </div>
                  } />
                </Routes>
              </ErrorBoundary>
            </main>
            <Footer />
          </div>
        </Router>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
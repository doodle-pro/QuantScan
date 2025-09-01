import React, { Component, ErrorInfo, ReactNode } from 'react';
import { ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-healthcare px-4">
          <div className="max-w-md w-full bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-comfortable border border-red-200 dark:border-red-700">
            <div className="text-center">
              <div className="w-16 h-16 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <ExclamationTriangleIcon className="w-8 h-8 text-red-600 dark:text-red-400" />
              </div>
              <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">
                Something went wrong
              </h2>
              <p className="text-slate-600 dark:text-slate-300 mb-6">
                We encountered an unexpected error. This might be a temporary issue.
              </p>
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-lg p-4 mb-6 text-left">
                  <p className="text-sm text-red-800 dark:text-red-300 font-mono">
                    {this.state.error.message}
                  </p>
                </div>
              )}
              <div className="space-y-3">
                <button
                  onClick={this.handleRetry}
                  className="btn-primary w-full inline-flex items-center justify-center space-x-2"
                >
                  <ArrowPathIcon className="w-5 h-5" />
                  <span>Try Again</span>
                </button>
                <button
                  onClick={() => window.location.href = '/'}
                  className="btn-soft w-full"
                >
                  Go to Home
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
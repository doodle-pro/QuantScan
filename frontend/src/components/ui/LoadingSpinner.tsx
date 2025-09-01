import React from 'react';
import { CpuChipIcon } from '@heroicons/react/24/outline';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  message?: string;
  quantum?: boolean;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  message = 'Loading...', 
  quantum = false 
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-4" role="status" aria-live="polite">
      <div className="relative">
        {quantum ? (
          <div className="relative">
            <CpuChipIcon className={`${sizeClasses[size]} text-blue-500 dark:text-blue-400 animate-pulse`} />
            <div className={`absolute inset-0 ${sizeClasses[size]} border-2 border-blue-300 dark:border-blue-600 border-t-blue-600 dark:border-t-blue-400 rounded-full animate-spin`} />
          </div>
        ) : (
          <div className={`${sizeClasses[size]} border-4 border-blue-100 dark:border-blue-800 border-t-blue-500 dark:border-t-blue-400 rounded-full animate-spin`} />
        )}
      </div>
      {message && (
        <p className={`${textSizeClasses[size]} text-slate-600 dark:text-slate-300 font-medium`}>
          {message}
        </p>
      )}
      <span className="sr-only">Loading content, please wait</span>
    </div>
  );
};

export default LoadingSpinner;
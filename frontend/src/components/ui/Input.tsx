import React, { InputHTMLAttributes, forwardRef } from 'react';
import { ExclamationCircleIcon, CheckCircleIcon } from '@heroicons/react/24/outline';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  success?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  required?: boolean;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ 
    className = '', 
    label, 
    error, 
    success, 
    helperText, 
    leftIcon, 
    rightIcon, 
    required,
    id,
    ...props 
  }, ref) => {
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const hasError = !!error;
    const hasSuccess = !!success;

    const baseClasses = 'w-full px-4 py-3 rounded-xl border bg-white/90 dark:bg-slate-800/90 dark:text-slate-200 dark:placeholder-slate-400 backdrop-blur-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1 dark:focus:ring-offset-slate-800';
    
    const stateClasses = hasError 
      ? 'border-red-300 dark:border-red-600 focus:border-red-400 dark:focus:border-red-500 focus:ring-red-100 dark:focus:ring-red-500/20' 
      : hasSuccess
      ? 'border-green-300 dark:border-green-600 focus:border-green-400 dark:focus:border-green-500 focus:ring-green-100 dark:focus:ring-green-500/20'
      : 'border-slate-200 dark:border-slate-600 focus:border-blue-300 dark:focus:border-blue-400 focus:ring-blue-100 dark:focus:ring-blue-500/20 hover:border-slate-300 dark:hover:border-slate-500';

    const iconClasses = hasError 
      ? 'text-red-500 dark:text-red-400' 
      : hasSuccess 
      ? 'text-green-500 dark:text-green-400' 
      : 'text-slate-400 dark:text-slate-500';

    return (
      <div className="space-y-2">
        {label && (
          <label 
            htmlFor={inputId}
            className="block text-sm font-medium text-slate-700 dark:text-slate-200"
          >
            {label}
            {required && <span className="text-red-500 dark:text-red-400 ml-1" aria-label="required">*</span>}
          </label>
        )}
        
        <div className="relative">
          {leftIcon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <span className={`w-5 h-5 ${iconClasses}`}>
                {leftIcon}
              </span>
            </div>
          )}
          
          <input
            ref={ref}
            id={inputId}
            className={`${baseClasses} ${stateClasses} ${leftIcon ? 'pl-10' : ''} ${rightIcon || hasError || hasSuccess ? 'pr-10' : ''} ${className}`}
            aria-invalid={hasError}
            aria-describedby={
              error ? `${inputId}-error` : 
              success ? `${inputId}-success` : 
              helperText ? `${inputId}-helper` : undefined
            }
            {...props}
          />
          
          {(rightIcon || hasError || hasSuccess) && (
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
              {hasError ? (
                <ExclamationCircleIcon className="w-5 h-5 text-red-500 dark:text-red-400" aria-hidden="true" />
              ) : hasSuccess ? (
                <CheckCircleIcon className="w-5 h-5 text-green-500 dark:text-green-400" aria-hidden="true" />
              ) : rightIcon ? (
                <span className={`w-5 h-5 ${iconClasses}`}>
                  {rightIcon}
                </span>
              ) : null}
            </div>
          )}
        </div>
        
        {error && (
          <p id={`${inputId}-error`} className="text-sm text-red-600 dark:text-red-400" role="alert">
            {error}
          </p>
        )}
        
        {success && !error && (
          <p id={`${inputId}-success`} className="text-sm text-green-600 dark:text-green-400">
            {success}
          </p>
        )}
        
        {helperText && !error && !success && (
          <p id={`${inputId}-helper`} className="text-sm text-slate-500 dark:text-slate-400">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
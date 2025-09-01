import React, { ButtonHTMLAttributes, forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import LoadingSpinner from './LoadingSpinner';

const buttonVariants = cva(
  'inline-flex items-center justify-center font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
  {
    variants: {
      variant: {
        primary: 'bg-gradient-to-r from-blue-500 to-teal-500 text-white hover:from-blue-600 hover:to-teal-600 focus:ring-blue-300 shadow-blue hover:shadow-large hover:scale-[1.02]',
        secondary: 'bg-gradient-to-r from-teal-500 to-green-500 text-white hover:from-teal-600 hover:to-green-600 focus:ring-teal-300 shadow-teal hover:shadow-large',
        accent: 'bg-gradient-to-r from-pink-300 to-pink-400 text-slate-700 hover:from-pink-400 hover:to-pink-500 focus:ring-pink-300 shadow-pink hover:shadow-large',
        outline: 'border-2 border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400 focus:ring-blue-300',
        soft: 'bg-white text-blue-600 border border-blue-200 hover:bg-blue-50 hover:border-blue-300 focus:ring-blue-300 shadow-soft hover:shadow-medium',
        ghost: 'text-slate-600 hover:text-blue-600 hover:bg-blue-50 focus:ring-blue-300',
        danger: 'bg-gradient-to-r from-red-500 to-red-600 text-white hover:from-red-600 hover:to-red-700 focus:ring-red-300 shadow-soft hover:shadow-medium'
      },
      size: {
        sm: 'px-3 py-2 text-sm rounded-lg',
        md: 'px-6 py-3 text-base rounded-xl',
        lg: 'px-8 py-4 text-lg rounded-xl',
        icon: 'p-2 rounded-lg'
      }
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md'
    }
  }
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
  loadingText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant, 
    size, 
    loading = false, 
    loadingText, 
    leftIcon, 
    rightIcon, 
    children, 
    disabled,
    ...props 
  }, ref) => {
    const isDisabled = disabled || loading;

    return (
      <button
        className={buttonVariants({ variant, size, className })}
        ref={ref}
        disabled={isDisabled}
        aria-disabled={isDisabled}
        {...props}
      >
        {loading ? (
          <>
            <LoadingSpinner size="sm" />
            {loadingText && <span className="ml-2">{loadingText}</span>}
          </>
        ) : (
          <>
            {leftIcon && <span className="mr-2">{leftIcon}</span>}
            {children}
            {rightIcon && <span className="ml-2">{rightIcon}</span>}
          </>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;
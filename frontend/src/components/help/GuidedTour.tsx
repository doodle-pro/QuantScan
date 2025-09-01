import React, { useState, useEffect, useRef } from 'react';
import { 
  XMarkIcon, 
  ArrowRightIcon, 
  ArrowLeftIcon,
  LightBulbIcon,
  CheckIcon
} from '@heroicons/react/24/outline';
import Button from '../ui/Button';

interface TourStep {
  target: string;
  title: string;
  content: string;
  position: 'top' | 'bottom' | 'left' | 'right';
  action?: () => void;
}

interface GuidedTourProps {
  steps: TourStep[];
  isActive: boolean;
  onComplete: () => void;
  onSkip: () => void;
}

const GuidedTour: React.FC<GuidedTourProps> = ({
  steps,
  isActive,
  onComplete,
  onSkip
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive) return;

    const updatePosition = () => {
      const step = steps[currentStep];
      if (!step) return;

      const targetElement = document.querySelector(step.target) as HTMLElement;
      if (!targetElement || !tooltipRef.current) return;

      const targetRect = targetElement.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const viewport = {
        width: window.innerWidth,
        height: window.innerHeight
      };

      let top = 0;
      let left = 0;

      switch (step.position) {
        case 'top':
          top = targetRect.top - tooltipRect.height - 16;
          left = targetRect.left + (targetRect.width - tooltipRect.width) / 2;
          break;
        case 'bottom':
          top = targetRect.bottom + 16;
          left = targetRect.left + (targetRect.width - tooltipRect.width) / 2;
          break;
        case 'left':
          top = targetRect.top + (targetRect.height - tooltipRect.height) / 2;
          left = targetRect.left - tooltipRect.width - 16;
          break;
        case 'right':
          top = targetRect.top + (targetRect.height - tooltipRect.height) / 2;
          left = targetRect.right + 16;
          break;
      }

      // Adjust for viewport boundaries
      if (left < 16) left = 16;
      if (left + tooltipRect.width > viewport.width - 16) {
        left = viewport.width - tooltipRect.width - 16;
      }
      if (top < 16) top = 16;
      if (top + tooltipRect.height > viewport.height - 16) {
        top = viewport.height - tooltipRect.height - 16;
      }

      setTooltipPosition({ top, left });

      // Highlight target element
      targetElement.style.position = 'relative';
      targetElement.style.zIndex = '1001';
      targetElement.style.boxShadow = '0 0 0 4px rgba(59, 130, 246, 0.5), 0 0 0 8px rgba(59, 130, 246, 0.2)';
      targetElement.style.borderRadius = '8px';

      // Scroll target into view
      targetElement.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center',
        inline: 'center'
      });
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition);

    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition);
      
      // Remove highlighting from all elements
      const highlightedElements = document.querySelectorAll('[style*="z-index: 1001"]');
      highlightedElements.forEach(el => {
        const element = el as HTMLElement;
        element.style.position = '';
        element.style.zIndex = '';
        element.style.boxShadow = '';
        element.style.borderRadius = '';
      });
    };
  }, [currentStep, steps, isActive]);

  useEffect(() => {
    if (!isActive) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onSkip();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isActive, onSkip]);

  const nextStep = () => {
    const step = steps[currentStep];
    step.action?.();

    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  if (!isActive || !steps[currentStep]) return null;

  const step = steps[currentStep];

  return (
    <>
      {/* Overlay */}
      <div
        ref={overlayRef}
        className="fixed inset-0 bg-black/50 z-1000"
        style={{ zIndex: 1000 }}
        onClick={onSkip}
      />

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className="fixed z-1001 bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-600 max-w-sm"
        style={{
          top: tooltipPosition.top,
          left: tooltipPosition.left,
          zIndex: 1001
        }}
        role="dialog"
        aria-labelledby="tour-title"
        aria-describedby="tour-content"
      >
        <div className="p-6">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
                <LightBulbIcon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 id="tour-title" className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                {step.title}
              </h3>
            </div>
            <button
              onClick={onSkip}
              className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
              aria-label="Close tour"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <p id="tour-content" className="text-slate-600 dark:text-slate-300 mb-6 leading-relaxed">
            {step.content}
          </p>

          {/* Progress */}
          <div className="mb-6">
            <div className="flex justify-between text-sm text-slate-500 dark:text-slate-400 mb-2">
              <span>Step {currentStep + 1} of {steps.length}</span>
              <span>{Math.round(((currentStep + 1) / steps.length) * 100)}%</span>
            </div>
            <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2">
              <div
                className="bg-blue-500 dark:bg-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-between items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={onSkip}
            >
              Skip Tour
            </Button>

            <div className="flex space-x-2">
              {currentStep > 0 && (
                <Button
                  variant="soft"
                  size="sm"
                  onClick={prevStep}
                  leftIcon={<ArrowLeftIcon className="w-4 h-4" />}
                >
                  Back
                </Button>
              )}
              <Button
                size="sm"
                onClick={nextStep}
                rightIcon={
                  currentStep === steps.length - 1 ? 
                    <CheckIcon className="w-4 h-4" /> : 
                    <ArrowRightIcon className="w-4 h-4" />
                }
              >
                {currentStep === steps.length - 1 ? 'Finish' : 'Next'}
              </Button>
            </div>
          </div>
        </div>

        {/* Arrow pointer */}
        <div
          className={`absolute w-3 h-3 bg-white dark:bg-slate-800 border-l border-t border-slate-200 dark:border-slate-600 transform rotate-45 ${
            step.position === 'top' ? 'bottom-[-6px] left-1/2 -translate-x-1/2' :
            step.position === 'bottom' ? 'top-[-6px] left-1/2 -translate-x-1/2' :
            step.position === 'left' ? 'right-[-6px] top-1/2 -translate-y-1/2' :
            'left-[-6px] top-1/2 -translate-y-1/2'
          }`}
        />
      </div>
    </>
  );
};

export default GuidedTour;
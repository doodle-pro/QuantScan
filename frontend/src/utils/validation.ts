// Form validation utilities

export interface ValidationRule {
  required?: boolean;
  min?: number;
  max?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  custom?: (value: any) => string | null;
}

export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
}

export class FormValidator {
  private rules: Record<string, ValidationRule> = {};
  private values: Record<string, any> = {};

  setRules(rules: Record<string, ValidationRule>) {
    this.rules = rules;
    return this;
  }

  setValues(values: Record<string, any>) {
    this.values = values;
    return this;
  }

  validateField(fieldName: string, value: any): string | null {
    const rule = this.rules[fieldName];
    if (!rule) return null;

    // Required validation
    if (rule.required && (value === undefined || value === null || value === '')) {
      return 'This field is required';
    }

    // Skip other validations if field is empty and not required
    if (!rule.required && (value === undefined || value === null || value === '')) {
      return null;
    }

    // Numeric validations
    if (typeof value === 'number') {
      if (rule.min !== undefined && value < rule.min) {
        return `Value must be at least ${rule.min}`;
      }
      if (rule.max !== undefined && value > rule.max) {
        return `Value must be at most ${rule.max}`;
      }
    }

    // String validations
    if (typeof value === 'string') {
      if (rule.minLength !== undefined && value.length < rule.minLength) {
        return `Must be at least ${rule.minLength} characters`;
      }
      if (rule.maxLength !== undefined && value.length > rule.maxLength) {
        return `Must be at most ${rule.maxLength} characters`;
      }
      if (rule.pattern && !rule.pattern.test(value)) {
        return 'Invalid format';
      }
    }

    // Custom validation
    if (rule.custom) {
      return rule.custom(value);
    }

    return null;
  }

  validate(): ValidationResult {
    const errors: Record<string, string> = {};

    for (const fieldName in this.rules) {
      const error = this.validateField(fieldName, this.values[fieldName]);
      if (error) {
        errors[fieldName] = error;
      }
    }

    return {
      isValid: Object.keys(errors).length === 0,
      errors
    };
  }

  validateSingle(fieldName: string): string | null {
    return this.validateField(fieldName, this.values[fieldName]);
  }
}

// Predefined validation rules
export const validationRules = {
  required: { required: true },
  email: {
    required: true,
    pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    custom: (value: string) => {
      if (value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
        return 'Please enter a valid email address';
      }
      return null;
    }
  },
  age: {
    required: true,
    min: 18,
    max: 100,
    custom: (value: number) => {
      if (value && (!Number.isInteger(value) || value < 18 || value > 100)) {
        return 'Age must be between 18 and 100';
      }
      return null;
    }
  },
  biomarker: {
    required: true,
    min: 0,
    custom: (value: number) => {
      if (value !== undefined && value !== null && (isNaN(value) || value < 0)) {
        return 'Must be a positive number';
      }
      return null;
    }
  },
  name: {
    minLength: 2,
    maxLength: 50,
    pattern: /^[a-zA-Z\s]+$/,
    custom: (value: string) => {
      if (value && !/^[a-zA-Z\s]+$/.test(value)) {
        return 'Name can only contain letters and spaces';
      }
      return null;
    }
  }
};

// Sanitization utilities
export const sanitizeInput = (input: string): string => {
  return input
    .trim()
    .replace(/[<>]/g, '') // Remove potential HTML tags
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, ''); // Remove event handlers
};

export const sanitizeNumber = (input: string | number): number => {
  if (typeof input === 'number') return input;
  const num = parseFloat(input.toString().replace(/[^0-9.-]/g, ''));
  return isNaN(num) ? 0 : num;
};

// Form state management
export interface FormState<T> {
  values: T;
  errors: Record<keyof T, string>;
  touched: Record<keyof T, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

export const createFormState = <T extends Record<string, any>>(
  initialValues: T
): FormState<T> => ({
  values: initialValues,
  errors: {} as Record<keyof T, string>,
  touched: {} as Record<keyof T, boolean>,
  isSubmitting: false,
  isValid: false
});

// Debounced validation hook utility
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

// Progress saving utilities
export const saveFormProgress = <T>(formId: string, values: T): void => {
  try {
    localStorage.setItem(`form_progress_${formId}`, JSON.stringify(values));
  } catch (error) {
    console.warn('Failed to save form progress:', error);
  }
};

export const loadFormProgress = <T>(formId: string): T | null => {
  try {
    const saved = localStorage.getItem(`form_progress_${formId}`);
    return saved ? JSON.parse(saved) : null;
  } catch (error) {
    console.warn('Failed to load form progress:', error);
    return null;
  }
};

export const clearFormProgress = (formId: string): void => {
  try {
    localStorage.removeItem(`form_progress_${formId}`);
  } catch (error) {
    console.warn('Failed to clear form progress:', error);
  }
};

// Accessibility helpers for forms
export const announceFormError = (fieldName: string, error: string): void => {
  const announcement = `Error in ${fieldName}: ${error}`;
  const element = document.createElement('div');
  element.setAttribute('aria-live', 'assertive');
  element.setAttribute('aria-atomic', 'true');
  element.className = 'sr-only';
  element.textContent = announcement;
  
  document.body.appendChild(element);
  setTimeout(() => document.body.removeChild(element), 1000);
};

export const focusFirstError = (formElement: HTMLElement): void => {
  const firstErrorField = formElement.querySelector('[aria-invalid="true"]') as HTMLElement;
  if (firstErrorField) {
    firstErrorField.focus();
    firstErrorField.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
};
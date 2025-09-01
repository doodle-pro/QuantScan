import { useState, useCallback } from 'react';
import { apiService, ApiResponse } from '../services/api';

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  success: boolean;
}

interface UseApiReturn<T> extends UseApiState<T> {
  execute: (...args: any[]) => Promise<T | null>;
  reset: () => void;
  retry: () => Promise<T | null>;
}

export const useApi = <T = any>(
  apiFunction: (...args: any[]) => Promise<ApiResponse<T>>,
  options: {
    immediate?: boolean;
    onSuccess?: (data: T) => void;
    onError?: (error: string) => void;
  } = {}
): UseApiReturn<T> => {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
    success: false
  });

  const [lastArgs, setLastArgs] = useState<any[]>([]);

  const execute = useCallback(async (...args: any[]): Promise<T | null> => {
    setState(prev => ({ ...prev, loading: true, error: null, success: false }));
    setLastArgs(args);

    try {
      const response = await apiFunction(...args);
      
      if (response.error) {
        setState(prev => ({ 
          ...prev, 
          loading: false, 
          error: response.error!, 
          success: false 
        }));
        options.onError?.(response.error);
        return null;
      }

      setState(prev => ({ 
        ...prev, 
        data: response.data!, 
        loading: false, 
        error: null, 
        success: true 
      }));
      options.onSuccess?.(response.data!);
      return response.data!;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: errorMessage, 
        success: false 
      }));
      options.onError?.(errorMessage);
      return null;
    }
  }, [apiFunction, options]);

  const retry = useCallback(async (): Promise<T | null> => {
    return execute(...lastArgs);
  }, [execute, lastArgs]);

  const reset = useCallback(() => {
    setState({
      data: null,
      loading: false,
      error: null,
      success: false
    });
  }, []);

  return {
    ...state,
    execute,
    retry,
    reset
  };
};

// Specific hooks for our API endpoints
export const usePrediction = (options?: {
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
}) => {
  return useApi(
    (patientData: any, useQuantum: boolean = true) => 
      apiService.predict(patientData, useQuantum),
    options
  );
};

export const useHealthCheck = (options?: {
  onSuccess?: (data: { status: string }) => void;
  onError?: (error: string) => void;
}) => {
  return useApi(
    () => apiService.healthCheck(),
    options
  );
};

// Hook for handling offline/online status
export const useOnlineStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useState(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  });

  return isOnline;
};

// Hook for managing request timeouts
export const useTimeout = (callback: () => void, delay: number | null) => {
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  const start = useCallback(() => {
    if (delay !== null) {
      const id = setTimeout(callback, delay);
      setTimeoutId(id);
      return id;
    }
  }, [callback, delay]);

  const clear = useCallback(() => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
  }, [timeoutId]);

  const restart = useCallback(() => {
    clear();
    return start();
  }, [clear, start]);

  return { start, clear, restart };
};
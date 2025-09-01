/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    screens: {
      'xs': '475px',
      'sm': '640px',
      'md': '768px',
      'lg': '1024px',
      'xl': '1280px',
      '2xl': '1536px',
    },
    extend: {
      colors: {
        // Soft & Aesthetic Healthcare Theme
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',  // Soft Blue
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#0F286E',  // Deep Blue for navigation
        },
        secondary: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',  // Soft Teal
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        accent: {
          50: '#fdf2f8',
          100: '#fce7f3',
          200: '#fbcfe8',
          300: '#f9a8d4',
          400: '#f472b6',
          500: '#ec4899',  // Soft Pink
          600: '#db2777',
          700: '#be185d',
          800: '#9d174d',
          900: '#831843',
        },
        // Soft Healthcare Colors
        healthcare: {
          // Soft Blues
          'blue-light': '#00BFFF',
          'blue-medium': '#4E7665',
          'blue-deep': '#0F286E',
          
          // Soft Pinks
          'pink-lightest': '#FDECEF',
          'pink-light': '#F6C8D1',
          'pink-medium': '#E99DAC',
          
          // Soft Greens
          'green-sage': '#47D69D',
          'green-soft': '#86efac',
          
          // Neutrals
          'neutral-warm': '#f8fafc',
          'neutral-light': '#f1f5f9',
          'neutral-medium': '#e2e8f0',
          'neutral-dark': '#64748b',
        },
        // Redefined color system for comfort
        blue: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',  // Main Blue
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0F286E',  // Deep Blue
          950: '#0c1e3d',  // Darker Blue for dark mode
        },
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',  // Darkest slate for dark mode
        },
        pink: {
          50: '#FDECEF',   // Lightest Pink
          100: '#F6C8D1',  // Light Pink
          200: '#f3e8ff',
          300: '#E99DAC',  // Medium Pink
          400: '#f472b6',
          500: '#ec4899',
          600: '#db2777',
          700: '#be185d',
          800: '#9d174d',
          900: '#831843',
        },
        teal: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',  // Main Teal
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        green: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#47D69D',  // Soft Sage Green
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        // Soft accent colors
        orange: {
          50: '#fff7ed',
          100: '#ffedd5',
          200: '#fed7aa',
          300: '#fdba74',
          400: '#fb923c',
          500: '#f97316',  // Soft Orange accent
          600: '#ea580c',
          700: '#c2410c',
          800: '#9a3412',
          900: '#7c2d12',
        },
        yellow: {
          50: '#fefce8',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',  // Soft Yellow accent
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
        // Utility colors (softer versions)
        success: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        warning: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
        error: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
        },
      },
      backgroundImage: {
        // Soft, comfortable gradients
        'gradient-primary': 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #bae6fd 100%)',
        'gradient-secondary': 'linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 50%, #99f6e4 100%)',
        'gradient-accent': 'linear-gradient(135deg, #FDECEF 0%, #F6C8D1 50%, #E99DAC 100%)',
        'gradient-hero': 'linear-gradient(135deg, #ffffff 0%, #f0f9ff 25%, #FDECEF 50%, #f0fdfa 75%, #ffffff 100%)',
        'gradient-card': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f0f9ff 100%)',
        'gradient-button': 'linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%)',
        'gradient-button-hover': 'linear-gradient(135deg, #0284c7 0%, #0d9488 100%)',
        'gradient-soft-blue': 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
        'gradient-soft-pink': 'linear-gradient(135deg, #FDECEF 0%, #F6C8D1 100%)',
        'gradient-soft-teal': 'linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%)',
        'gradient-warm': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
        'gradient-healthcare': 'linear-gradient(135deg, #f0f9ff 0%, #FDECEF 25%, #f0fdfa 50%, #ffffff 100%)',
      },
      boxShadow: {
        'soft': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'medium': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'large': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'blue': '0 4px 14px 0 rgba(14, 165, 233, 0.15)',
        'teal': '0 4px 14px 0 rgba(20, 184, 166, 0.15)',
        'pink': '0 4px 14px 0 rgba(233, 157, 172, 0.15)',
        'comfortable': '0 2px 8px 0 rgba(15, 40, 110, 0.08)',
      },
      animation: {
        'fade-in': 'fadeIn 0.6s ease-out',
        'slide-up': 'slideUp 0.6s ease-out',
        'gentle-bounce': 'gentleBounce 3s infinite',
        'soft-pulse': 'softPulse 3s infinite',
        'gentle-float': 'gentleFloat 4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        gentleBounce: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-3px)' },
        },
        softPulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.9' },
        },
        gentleFloat: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-5px)' },
        },
      },
    },
  },
  plugins: [],
}
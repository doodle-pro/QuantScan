import React from 'react';
import { UserIcon, CalendarDaysIcon } from '@heroicons/react/24/outline';
import Input from '../ui/Input';
import Button from '../ui/Button';

interface PatientInfo {
  age: number;
  name: string;
  medicalHistory: string[];
}

interface PatientInfoFormProps {
  patientInfo: PatientInfo;
  setPatientInfo: React.Dispatch<React.SetStateAction<PatientInfo>>;
  onNext: () => void;
  onBack: () => void;
  inputMethod: 'manual' | 'sample' | 'upload';
  sampleCases?: Array<{
    name: string;
    description: string;
    patientInfo: PatientInfo;
    data: any;
  }>;
  selectedSample: number;
  onSampleSelection: (index: number) => void;
}

const PatientInfoForm: React.FC<PatientInfoFormProps> = ({
  patientInfo,
  setPatientInfo,
  onNext,
  onBack,
  inputMethod,
  sampleCases = [],
  selectedSample,
  onSampleSelection
}) => {
  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPatientInfo(prev => ({ ...prev, name: e.target.value }));
  };

  const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const age = parseInt(e.target.value) || 0;
    setPatientInfo(prev => ({ ...prev, age }));
  };

  const validateForm = () => {
    return patientInfo.age >= 18 && patientInfo.age <= 100;
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="healthcare-card">
        <div className="flex items-center space-x-3 mb-6">
          <UserIcon className="w-6 h-6 text-blue-500 dark:text-blue-400" />
          <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-100">
            Patient Information
          </h3>
        </div>
        
        {inputMethod === 'sample' && sampleCases.length > 0 && (
          <div className="mb-8">
            <h4 className="text-lg font-medium text-slate-800 dark:text-slate-100 mb-4">Select Sample Case</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {sampleCases.map((sample, index) => (
                <button
                  key={index}
                  onClick={() => onSampleSelection(index)}
                  className={`p-4 rounded-xl border-2 text-left transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-500 focus:ring-offset-1 dark:focus:ring-offset-slate-800 ${
                    selectedSample === index
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 shadow-blue'
                      : 'border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20'
                  }`}
                  aria-pressed={selectedSample === index}
                >
                  <h5 className="font-semibold text-slate-800 dark:text-slate-100 mb-2">{sample.name}</h5>
                  <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">{sample.description}</p>
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    <div>Age: {sample.patientInfo.age}</div>
                    <div>History: {sample.patientInfo.medicalHistory.join(', ')}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Input
            label="Patient Name (Optional)"
            type="text"
            value={patientInfo.name}
            onChange={handleNameChange}
            placeholder="Enter patient name"
            leftIcon={<UserIcon className="w-5 h-5" />}
          />
          <Input
            label="Age"
            type="number"
            value={patientInfo.age.toString()}
            onChange={handleAgeChange}
            placeholder="Enter age"
            min="18"
            max="100"
            required
            leftIcon={<CalendarDaysIcon className="w-5 h-5" />}
            error={patientInfo.age < 18 || patientInfo.age > 100 ? 'Age must be between 18 and 100' : undefined}
            helperText="Patient age is required for accurate risk assessment"
          />
        </div>

        <div className="mt-8 flex justify-between">
          <Button
            variant="soft"
            onClick={onBack}
            leftIcon={<span>←</span>}
          >
            Back
          </Button>
          <Button
            onClick={onNext}
            disabled={!validateForm()}
            rightIcon={<span>→</span>}
          >
            Continue to Biomarkers
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PatientInfoForm;
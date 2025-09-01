import React, { useState, useCallback } from 'react';
import { BeakerIcon, DocumentArrowUpIcon, InformationCircleIcon, CloudArrowUpIcon, XMarkIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import Input from '../ui/Input';
import Button from '../ui/Button';
import { InfoTooltip } from '../ui/Tooltip';

interface BiomarkerData {
  // Mean values
  mean_radius: number;
  mean_texture: number;
  mean_perimeter: number;
  mean_area: number;
  mean_smoothness: number;
  mean_compactness: number;
  mean_concavity: number;
  mean_concave_points: number;
  mean_symmetry: number;
  mean_fractal_dimension: number;
  
  // Standard error values
  radius_error: number;
  texture_error: number;
  perimeter_error: number;
  area_error: number;
  smoothness_error: number;
  compactness_error: number;
  concavity_error: number;
  concave_points_error: number;
  symmetry_error: number;
  fractal_dimension_error: number;
  
  // Worst values
  worst_radius: number;
  worst_texture: number;
  worst_perimeter: number;
  worst_area: number;
  worst_smoothness: number;
  worst_compactness: number;
  worst_concavity: number;
  worst_concave_points: number;
  worst_symmetry: number;
  worst_fractal_dimension: number;
}

interface BiomarkerFormProps {
  biomarkerData: BiomarkerData;
  setBiomarkerData: React.Dispatch<React.SetStateAction<BiomarkerData>>;
  onNext: () => void;
  onBack: () => void;
  inputMethod: 'manual' | 'sample' | 'upload';
  isAnalyzing?: boolean;
}

const BiomarkerForm: React.FC<BiomarkerFormProps> = ({
  biomarkerData,
  setBiomarkerData,
  onNext,
  onBack,
  inputMethod,
  isAnalyzing = false
}) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isProcessingFile, setIsProcessingFile] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const biomarkerCategories = [
    {
      name: "Cell Nucleus Measurements (Mean)",
      description: "Average measurements of cell nucleus characteristics",
      fields: [
        { key: 'mean_radius', label: 'Radius', unit: 'μm', description: 'Mean distance from center to perimeter' },
        { key: 'mean_texture', label: 'Texture', unit: '', description: 'Standard deviation of gray-scale values' },
        { key: 'mean_perimeter', label: 'Perimeter', unit: 'μm', description: 'Cell nucleus perimeter' },
        { key: 'mean_area', label: 'Area', unit: 'μm²', description: 'Cell nucleus area' },
        { key: 'mean_smoothness', label: 'Smoothness', unit: '', description: 'Local variation in radius lengths' },
        { key: 'mean_compactness', label: 'Compactness', unit: '', description: 'Perimeter² / area - 1.0' },
        { key: 'mean_concavity', label: 'Concavity', unit: '', description: 'Severity of concave portions' },
        { key: 'mean_concave_points', label: 'Concave Points', unit: '', description: 'Number of concave portions' },
        { key: 'mean_symmetry', label: 'Symmetry', unit: '', description: 'Cell nucleus symmetry' },
        { key: 'mean_fractal_dimension', label: 'Fractal Dimension', unit: '', description: 'Coastline approximation - 1' }
      ]
    },
    {
      name: "Standard Error Measurements",
      description: "Standard error of measurements across multiple samples",
      fields: [
        { key: 'radius_error', label: 'Radius Error', unit: 'μm', description: 'Standard error of radius' },
        { key: 'texture_error', label: 'Texture Error', unit: '', description: 'Standard error of texture' },
        { key: 'perimeter_error', label: 'Perimeter Error', unit: 'μm', description: 'Standard error of perimeter' },
        { key: 'area_error', label: 'Area Error', unit: 'μm²', description: 'Standard error of area' },
        { key: 'smoothness_error', label: 'Smoothness Error', unit: '', description: 'Standard error of smoothness' },
        { key: 'compactness_error', label: 'Compactness Error', unit: '', description: 'Standard error of compactness' },
        { key: 'concavity_error', label: 'Concavity Error', unit: '', description: 'Standard error of concavity' },
        { key: 'concave_points_error', label: 'Concave Points Error', unit: '', description: 'Standard error of concave points' },
        { key: 'symmetry_error', label: 'Symmetry Error', unit: '', description: 'Standard error of symmetry' },
        { key: 'fractal_dimension_error', label: 'Fractal Dimension Error', unit: '', description: 'Standard error of fractal dimension' }
      ]
    },
    {
      name: "Worst Case Measurements",
      description: "Worst (largest) values found in the sample",
      fields: [
        { key: 'worst_radius', label: 'Worst Radius', unit: 'μm', description: 'Largest radius measurement' },
        { key: 'worst_texture', label: 'Worst Texture', unit: '', description: 'Largest texture measurement' },
        { key: 'worst_perimeter', label: 'Worst Perimeter', unit: 'μm', description: 'Largest perimeter measurement' },
        { key: 'worst_area', label: 'Worst Area', unit: 'μm²', description: 'Largest area measurement' },
        { key: 'worst_smoothness', label: 'Worst Smoothness', unit: '', description: 'Largest smoothness measurement' },
        { key: 'worst_compactness', label: 'Worst Compactness', unit: '', description: 'Largest compactness measurement' },
        { key: 'worst_concavity', label: 'Worst Concavity', unit: '', description: 'Largest concavity measurement' },
        { key: 'worst_concave_points', label: 'Worst Concave Points', unit: '', description: 'Largest concave points measurement' },
        { key: 'worst_symmetry', label: 'Worst Symmetry', unit: '', description: 'Largest symmetry measurement' },
        { key: 'worst_fractal_dimension', label: 'Worst Fractal Dimension', unit: '', description: 'Largest fractal dimension measurement' }
      ]
    }
  ];

  // CSV parsing function
  const parseCSV = useCallback((csvText: string): BiomarkerData | null => {
    try {
      const lines = csvText.trim().split('\n');
      if (lines.length < 2) {
        throw new Error('CSV must have at least a header and one data row');
      }

      const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
      const values = lines[1].split(',').map(v => parseFloat(v.trim()));

      if (values.length !== 30) {
        throw new Error('CSV must contain exactly 30 biomarker values');
      }

      // Expected field order for UCI Breast Cancer dataset
      const expectedFields = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
        'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
      ];

      const parsedData: BiomarkerData = {} as BiomarkerData;
      
      // Map values to biomarker fields
      expectedFields.forEach((field, index) => {
        if (index < values.length && !isNaN(values[index])) {
          (parsedData as any)[field] = values[index];
        } else {
          throw new Error(`Invalid or missing value for ${field}`);
        }
      });

      // Validate ranges
      if (parsedData.mean_radius < 0 || parsedData.mean_radius > 50) {
        throw new Error('Mean radius must be between 0 and 50 μm');
      }
      if (parsedData.mean_area < 0 || parsedData.mean_area > 5000) {
        throw new Error('Mean area must be between 0 and 5000 μm²');
      }

      return parsedData;
    } catch (error) {
      console.error('CSV parsing error:', error);
      return null;
    }
  }, []);

  // Handle file upload
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsProcessingFile(true);
    setUploadError(null);
    setUploadSuccess(false);

    try {
      // Validate file type
      if (!file.name.toLowerCase().endsWith('.csv')) {
        throw new Error('Please upload a CSV file');
      }

      // Validate file size (max 1MB)
      if (file.size > 1024 * 1024) {
        throw new Error('File size must be less than 1MB');
      }

      // Read file content
      const text = await file.text();
      const parsedData = parseCSV(text);

      if (!parsedData) {
        throw new Error('Failed to parse CSV data. Please check the format.');
      }

      // Update biomarker data
      setBiomarkerData(parsedData);
      setUploadedFile(file);
      setUploadSuccess(true);

    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to process file');
    } finally {
      setIsProcessingFile(false);
    }
  }, [parseCSV, setBiomarkerData]);

  // Remove uploaded file
  const removeUploadedFile = useCallback(() => {
    setUploadedFile(null);
    setUploadSuccess(false);
    setUploadError(null);
    // Reset file input
    const fileInput = document.getElementById('csv-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  }, []);

  const handleInputChange = (field: keyof BiomarkerData, value: string) => {
    const numValue = parseFloat(value) || 0;
    setBiomarkerData(prev => ({
      ...prev,
      [field]: numValue
    }));
  };

  const validateForm = () => {
    // Check if all required fields have valid values
    const isValid = Object.values(biomarkerData).every(value => 
      typeof value === 'number' && !isNaN(value) && value >= 0
    );

    // For upload method, also check if file was successfully processed
    if (inputMethod === 'upload') {
      return isValid && uploadSuccess;
    }

    return isValid;
  };

  // Generate sample CSV for download
  const generateSampleCSV = useCallback(() => {
    const headers = [
      'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
      'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
      'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
      'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
      'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
      'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
    ];

    const sampleValues = [
      17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
      1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
      25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ];

    const csvContent = headers.join(',') + '\n' + sampleValues.join(',');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'sample_biomarker_data.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="healthcare-card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <BeakerIcon className="w-6 h-6 text-blue-500 dark:text-blue-400" />
            <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-100">
              Biomarker Data Entry
            </h3>
          </div>
          <div className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-300">
            <InformationCircleIcon className="w-5 h-5" />
            <span>30 biomarker measurements required</span>
          </div>
        </div>

        {inputMethod === 'upload' && (
          <div className="mb-8 space-y-6">
            {/* Upload Area */}
            <div className={`p-6 border-2 border-dashed rounded-xl text-center transition-all duration-200 ${
              uploadSuccess 
                ? 'border-green-300 dark:border-green-600 bg-green-50 dark:bg-green-900/30' 
                : uploadError 
                  ? 'border-red-300 dark:border-red-600 bg-red-50 dark:bg-red-900/30' 
                  : 'border-slate-300 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 hover:bg-blue-50/30 dark:hover:bg-blue-900/20'
            }`}>
              {isProcessingFile ? (
                <div className="space-y-4">
                  <CloudArrowUpIcon className="w-12 h-12 text-blue-500 dark:text-blue-400 mx-auto animate-pulse" />
                  <div className="text-lg font-medium text-slate-800 dark:text-slate-100">Processing File...</div>
                  <div className="text-sm text-slate-600 dark:text-slate-300">Validating biomarker data</div>
                </div>
              ) : uploadSuccess && uploadedFile ? (
                <div className="space-y-4">
                  <CheckCircleIcon className="w-12 h-12 text-green-500 dark:text-green-400 mx-auto" />
                  <div className="text-lg font-medium text-green-800 dark:text-green-300">File Uploaded Successfully!</div>
                  <div className="text-sm text-green-700 dark:text-green-400">
                    {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                  </div>
                  <div className="flex items-center justify-center space-x-4">
                    <Button
                      variant="soft"
                      size="sm"
                      onClick={removeUploadedFile}
                      leftIcon={<XMarkIcon className="w-4 h-4" />}
                    >
                      Remove File
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <DocumentArrowUpIcon className="w-12 h-12 text-slate-400 dark:text-slate-500 mx-auto" />
                  <div className="text-lg font-medium text-slate-800 dark:text-slate-100">Upload CSV File</div>
                  <div className="text-sm text-slate-600 dark:text-slate-300 mb-4">
                    Upload a CSV file with 30 biomarker values following the UCI Breast Cancer dataset format
                  </div>
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    id="csv-upload"
                    onChange={handleFileUpload}
                    aria-label="Upload CSV file with biomarker data"
                  />
                  <div className="flex items-center justify-center space-x-4">
                    <label
                      htmlFor="csv-upload"
                      className="btn-primary cursor-pointer inline-flex items-center space-x-2"
                    >
                      <DocumentArrowUpIcon className="w-5 h-5" />
                      <span>Choose CSV File</span>
                    </label>
                    <Button
                      variant="soft"
                      size="sm"
                      onClick={generateSampleCSV}
                      leftIcon={<DocumentArrowUpIcon className="w-4 h-4" />}
                    >
                      Download Sample
                    </Button>
                  </div>
                </div>
              )}
            </div>

            {/* Error Display */}
            {uploadError && (
              <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-xl">
                <div className="flex items-center space-x-2">
                  <XMarkIcon className="w-5 h-5 text-red-600 dark:text-red-400" />
                  <div className="text-red-700 dark:text-red-300 font-medium">Upload Error</div>
                </div>
                <div className="text-red-600 dark:text-red-400 text-sm mt-1">{uploadError}</div>
              </div>
            )}

            {/* CSV Format Instructions */}
            <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <InformationCircleIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <div className="text-blue-800 dark:text-blue-300 font-medium">CSV Format Requirements</div>
              </div>
              <div className="text-blue-700 dark:text-blue-400 text-sm space-y-1">
                <div>• File must contain exactly 30 biomarker values</div>
                <div>• Values should be in the order: mean values (10), error values (10), worst values (10)</div>
                <div>• First row can be headers (optional), data should be in the second row</div>
                <div>• Values must be numeric and within valid ranges</div>
                <div>• Maximum file size: 1MB</div>
              </div>
            </div>

            {/* Data Preview */}
            {uploadSuccess && (
              <div className="bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl p-4">
                <div className="text-slate-800 dark:text-slate-100 font-medium mb-3">Data Preview</div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-slate-600 dark:text-slate-300">Mean Radius:</div>
                    <div className="font-medium text-slate-800 dark:text-slate-100">{biomarkerData.mean_radius.toFixed(3)} μm</div>
                  </div>
                  <div>
                    <div className="text-slate-600 dark:text-slate-300">Mean Area:</div>
                    <div className="font-medium text-slate-800 dark:text-slate-100">{biomarkerData.mean_area.toFixed(1)} μm²</div>
                  </div>
                  <div>
                    <div className="text-slate-600 dark:text-slate-300">Worst Radius:</div>
                    <div className="font-medium text-slate-800 dark:text-slate-100">{biomarkerData.worst_radius.toFixed(3)} μm</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {(inputMethod === 'manual' || inputMethod === 'sample') && (
          <div className="space-y-8">
            {biomarkerCategories.map((category, categoryIndex) => (
              <div key={categoryIndex} className="bg-slate-50 dark:bg-slate-700 rounded-xl p-6">
                <div className="flex items-center space-x-2 mb-2">
                  <h4 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                    {category.name}
                  </h4>
                  <InfoTooltip content={category.description} />
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-300 mb-6">
                  {category.description}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {category.fields.map((field) => (
                    <div key={field.key}>
                      <Input
                        label={`${field.label} ${field.unit ? `(${field.unit})` : ''}`}
                        type="number"
                        step="any"
                        value={biomarkerData[field.key as keyof BiomarkerData].toString()}
                        onChange={(e) => handleInputChange(field.key as keyof BiomarkerData, e.target.value)}
                        disabled={inputMethod === 'sample'}
                        helperText={field.description}
                        className="text-sm"
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

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
            leftIcon={<BeakerIcon className="w-5 h-5" />}
          >
            Run Quantum Analysis
          </Button>
        </div>
      </div>
    </div>
  );
};

export default BiomarkerForm;
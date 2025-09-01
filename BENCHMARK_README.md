# Breast Cancer Screening Tool - Benchmarking Suite

## Overview

This benchmarking suite provides comprehensive evaluation of the AI-enhanced breast cancer screening tool using real-world clinical scenarios and performance metrics.

## Benchmarking Scripts

### 1. `benchmark.py` - Standard Benchmark
Comprehensive evaluation with 8 real-world test cases:

```bash
python benchmark.py
```

**Features:**
- Model performance evaluation (accuracy, precision, recall, F1-score, AUC-ROC)
- AI enhancement impact analysis
- Timing and performance measurement
- Real-world case validation
- Comparative analysis with baseline algorithms
- Automated PDF report generation with charts

### 2. `benchmark_extended.py` - Extended Benchmark
Enhanced evaluation with additional test cases and stress testing:

```bash
python benchmark_extended.py
```

**Additional Features:**
- 14 total test cases including challenging edge cases
- Stress testing with rapid successive predictions
- Extended real-world scenarios (early stage, inflammatory, atypical cases)
- JSON results export for detailed analysis

## Generated Outputs

### PDF Reports
- **Standard Report**: `breast_cancer_benchmark_report_[timestamp].pdf`
- **Extended Report**: `extended_benchmark_report_[timestamp].pdf`

### Visualization Charts (`benchmark_charts/` folder)
- `performance_metrics.png` - Model metrics, confusion matrix, ROC curve
- `timing_analysis.png` - Response time comparison and overhead breakdown
- `real_world_analysis.png` - Case-by-case accuracy and confidence analysis

### Data Exports
- `benchmark_results_[timestamp].json` - Complete benchmark results in JSON format

## Real-World Test Cases

The benchmark includes diverse clinical scenarios:

### Malignant Cases
- **MAL_001**: Large irregular mass with high concavity
- **MAL_002**: Small but aggressive malignant tumor  
- **MAL_003**: Irregular invasive carcinoma
- **MAL_004**: Inflammatory breast cancer pattern
- **MAL_EARLY_001**: Early stage invasive ductal carcinoma
- **MAL_INFLAMMATORY_001**: Inflammatory breast cancer
- **MAL_TNBC_001**: Triple negative breast cancer

### Benign Cases
- **BEN_001**: Small regular mass with smooth borders
- **BEN_002**: Large but benign mass (fibroadenoma)
- **BEN_003**: Complex sclerosing lesion
- **BEN_ATYPICAL_001**: Atypical fibroadenoma
- **BEN_LIPOMA_001**: Classic breast lipoma
- **BEN_POSTSURG_001**: Post-surgical fibrotic changes

### Challenging Cases
- **BOR_001**: Borderline case with intermediate features

## Key Metrics Evaluated

### Performance Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: Malignant prediction reliability (reduces false positives)
- **Recall**: Malignant detection rate (reduces false negatives)
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Model discrimination ability

### AI Enhancement Analysis
- Base model vs AI-enhanced accuracy comparison
- Confidence score evaluation
- Medical knowledge integration impact

### Timing Analysis
- Base model prediction speed
- Enhanced prediction response time
- RAG knowledge retrieval performance
- Overall system overhead

## Technical Details

### Model Architecture
- **Base Model**: Random Forest Classifier (trained on Wisconsin Breast Cancer Dataset)
- **AI Enhancement**: Groq Llama3-8B model integration
- **Knowledge Base**: RAG with ChromaDB vector storage
- **Embedding Model**: all-MiniLM-L6-v2 (sentence-transformers)

### Feature Space
The benchmark uses 15 key tumor characteristics:
- `radius_worst`, `perimeter_worst`, `area_worst`, `concave_points_worst`
- `concavity_worst`, `compactness_worst`, `radius_mean`, `perimeter_mean`
- `area_mean`, `concave_points_mean`, `concavity_mean`, `compactness_mean`
- `texture_worst`, `smoothness_worst`, `symmetry_worst`

## Requirements

```bash
pip install reportlab seaborn matplotlib scikit-learn numpy pandas
```

## Usage Examples

### Quick Benchmark
```bash
# Run standard benchmark
python benchmark.py

# Check generated files
ls *.pdf
ls benchmark_charts/
```

### Extended Analysis
```bash
# Run extended benchmark with stress testing
python benchmark_extended.py

# View JSON results
cat benchmark_results_*.json | python -m json.tool
```

## Interpreting Results

### Performance Scores
- **0.90+**: Excellent performance
- **0.80-0.89**: Good performance  
- **0.70-0.79**: Acceptable performance
- **<0.70**: Needs improvement

### Timing Benchmarks
- **Base Model**: Should be <0.1s for real-time applications
- **Enhanced Prediction**: Includes AI analysis overhead (~10-15s typical)
- **RAG Retrieval**: Medical knowledge lookup (~3-5s typical)

### Clinical Validation
Real-world case validation demonstrates performance on actual medical scenarios with known ground truth diagnoses.

## Important Notes

⚠️ **Medical Disclaimer**: This benchmarking tool is for research and development purposes. Clinical decisions should never be based solely on these predictions without proper medical consultation and tissue biopsy confirmation.

⚠️ **Data Privacy**: All test cases use synthetic data based on medical literature, not actual patient data.

⚠️ **Performance Variance**: Results may vary based on API response times, system resources, and network connectivity.

# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an AI-enhanced breast cancer prediction system that combines traditional machine learning with modern AI services and RAG (Retrieval-Augmented Generation) for improved accuracy and medical interpretability. The system provides both basic ML predictions and comprehensive AI-powered medical analysis.

## Technology Stack

### Core Technologies
- **Python 3.11** - Primary language
- **FastAPI** - REST API framework
- **Streamlit** - Web UI framework
- **CLI Application** - Rich terminal interface with file processing
- **scikit-learn** - Machine learning (Random Forest, Decision Tree, SVM, KNN models tested)
- **joblib** - Model serialization
- **pandas/numpy** - Data processing
- **Jupyter Notebook** - Model development and experimentation

### AI Services
- **Groq API** - Primary AI provider (Llama3-8B model)
- **HuggingFace API** - Secondary AI provider (BioGPT model)
- **sentence-transformers** - Text embeddings (all-MiniLM-L6-v2)
- **ChromaDB** - Vector database for RAG
- **LangChain** - RAG framework and utilities

### Environment Management
- **python-dotenv** - Environment variable management
- **loguru** - Enhanced logging

## Architecture

The system follows a multi-layered AI-enhanced architecture:

### Core Components

1. **`main.py`** - Enhanced FastAPI backend server
   - Integrates traditional ML model with AI services
   - Provides both basic (`/predict`) and enhanced (`/predict/enhanced`) endpoints
   - Service status monitoring (`/status`, `/health`)
   - Comprehensive error handling and fallback mechanisms

2. **`app_ui.py`** - AI-powered Streamlit interface
   - Dual prediction modes (Basic ML vs AI-Enhanced)
   - Real-time service status monitoring
   - Tabbed results with detailed AI analysis
   - Risk assessment visualization

3. **`cli_app.py`** - Rich terminal CLI application
   - Interactive conversation with AI assistant
   - File processing (PDF, Word, TXT, JPEG, PNG)
   - Context-aware conversation history
   - Medical image and document analysis
   - Beautiful terminal interface with Rich library

3. **`services/`** - AI and RAG service layer
   - `enhanced_prediction_service.py` - Main orchestration service
   - `groq_service.py` - Primary AI provider (Llama3-8B)
   - `huggingface_service.py` - Secondary AI provider (BioGPT)
   - `rag_service.py` - Medical knowledge retrieval system

4. **`rag_data/`** - Knowledge management
   - `medical_knowledge.txt` - Curated medical knowledge base
   - `vector_store/` - ChromaDB vector database

5. **`.env`** - Secure environment configuration
   - API keys for Groq and HuggingFace
   - Model and service configuration

### Enhanced Data Flow
1. **Basic Mode**: User input → Streamlit → FastAPI → Random Forest → Simple prediction
2. **Enhanced Mode**: User input → Streamlit → FastAPI → {
   - Random Forest prediction
   - RAG knowledge retrieval
   - Groq AI analysis (primary)
   - HuggingFace AI analysis (fallback)
   - Risk assessment computation
   } → Comprehensive medical analysis

### AI Service Architecture
- **Primary AI Provider**: Groq (Llama3-8B) - Fast, reliable inference
- **Secondary AI Provider**: HuggingFace (BioGPT) - Medical domain specialization
- **Fallback Strategy**: Graceful degradation to basic ML if AI services fail
- **RAG Enhancement**: Context-aware medical knowledge retrieval

## Development Commands

### Environment Setup
```powershell
# Activate virtual environment (Windows)
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```powershell
# Start FastAPI backend server
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Start Streamlit web interface (in separate terminal)
streamlit run app_ui.py

# Start CLI application
python run_cli.py

# Or run CLI directly with options
python cli_app.py --status              # Check service status
python cli_app.py --predict             # Interactive prediction mode
python cli_app.py --file report.pdf     # Process specific file
```

### AI Services Setup
```powershell
# Set up API keys (add to .env file)
# GROQ_API_KEY=your_groq_key_here
# HUGGINGFACE_API_TOKEN=your_hf_token_here

# Initialize RAG knowledge base (first time only)
# The RAG service auto-initializes on first run

# Check service status
curl http://127.0.0.1:8000/status

# Detailed health check
curl http://127.0.0.1:8000/health
```

### Development Workflow
```powershell
# Run Jupyter notebook for model experimentation
jupyter notebook cancer_prediction.ipynb

# Test basic prediction endpoint
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "radius_worst": 16.0,
  "perimeter_worst": 100.0,
  "area_worst": 800.0,
  "concave_points_worst": 0.2,
  "concavity_worst": 0.3,
  "compactness_worst": 0.2,
  "radius_mean": 14.0,
  "perimeter_mean": 90.0,
  "area_mean": 700.0,
  "concave_points_mean": 0.1,
  "concavity_mean": 0.2,
  "compactness_mean": 0.1,
  "texture_worst": 20.0,
  "smoothness_worst": 0.1,
  "symmetry_worst": 0.3
}'

# Test enhanced AI prediction endpoint
curl -X POST "http://127.0.0.1:8000/predict/enhanced" -H "Content-Type: application/json" -d '{
  "radius_worst": 25.0,
  "perimeter_worst": 150.0,
  "area_worst": 1800.0,
  "concave_points_worst": 0.25,
  "concavity_worst": 0.4,
  "compactness_worst": 0.3,
  "radius_mean": 18.0,
  "perimeter_mean": 120.0,
  "area_mean": 1200.0,
  "concave_points_mean": 0.15,
  "concavity_mean": 0.25,
  "compactness_mean": 0.2,
  "texture_worst": 28.0,
  "smoothness_worst": 0.12,
  "symmetry_worst": 0.35
}'

# Check API documentation
# Navigate to http://127.0.0.1:8000/docs when FastAPI server is running
```

### Model Development
```powershell
# Re-train model (within Jupyter notebook environment)
# 1. Load and preprocess Cancer_Data.csv
# 2. Run model comparison cells
# 3. Export new model: joblib.dump(model, "cancer_model.pkl")
```

## Key Features

### Model Input Features (15 selected features)
The model uses these tumor characteristics for prediction:
- **Worst measurements**: radius, perimeter, area, concave_points, concavity, compactness, texture, smoothness, symmetry
- **Mean measurements**: radius, perimeter, area, concave_points, concavity, compactness

### API Endpoints
- `GET /` - Root health check
- `GET /status` - Service status information
- `GET /health` - Detailed health check with service diagnostics
- `POST /predict` - Basic tumor classification (ML only)
- `POST /predict/enhanced` - AI-enhanced analysis with RAG

### Development Notes

- The project requires the dataset file `Cancer_Data.csv` for model retraining (not included in repo)
- AI services require valid API keys for Groq and HuggingFace in the `.env` file
- Model achieves ~95% cross-validation accuracy with Random Forest
- Graceful degradation: System falls back to ML-only mode if AI services are unavailable
- ChromaDB creates and maintains vector store on first run
- Medical knowledge is auto-indexed on system startup
- All 15 input features are required for prediction
- Streamlit UI runs on default port (8501) and connects to FastAPI on port 8000
- Virtual environment is pre-configured for Python 3.11

### Feature Engineering Context
The 15 features were selected as the most important from the original 30 features in the Wisconsin Breast Cancer Dataset through feature importance analysis in the Random Forest model.

## AI Models and Licensing

### Selected Models (Apache 2.0 Licensed)

**Primary: Groq - Llama3-8B-8192**
- **License**: Apache 2.0
- **Strengths**: Fast inference, reliable performance, general medical knowledge
- **Use Case**: Primary AI analysis and medical interpretation
- **Provider**: Groq API (high-performance inference)

**Secondary: HuggingFace - BioGPT-Large**
- **License**: Apache 2.0  
- **Strengths**: Biomedical domain specialization, scientific text understanding
- **Use Case**: Fallback AI analysis when Groq is unavailable
- **Provider**: HuggingFace Inference API

**Embeddings: all-MiniLM-L6-v2**
- **License**: Apache 2.0
- **Strengths**: Efficient semantic search, good medical text representation
- **Use Case**: RAG knowledge retrieval and vector similarity search

### RAG System Architecture

**Knowledge Base Components:**
- Tumor characteristic definitions and normal ranges
- Diagnostic patterns for benign vs malignant tumors
- Clinical decision factors and risk indicators
- Feature importance rankings and interpretations
- Medical recommendations based on prediction outcomes

**Vector Database (ChromaDB):**
- Persistent vector storage for medical knowledge
- Semantic search capabilities for context retrieval
- Automatic knowledge base initialization
- Efficient similarity matching for relevant information

**Knowledge Retrieval Process:**
1. Analyze input tumor features
2. Generate semantic search query based on feature values
3. Retrieve most relevant medical knowledge chunks
4. Provide context to AI models for enhanced analysis

### AI Service Integration Benefits

**Enhanced Accuracy:**
- Combines statistical ML confidence with AI reasoning
- Medical domain knowledge improves interpretation quality
- Feature-specific insights help identify concerning patterns

**Improved Interpretability:**
- Natural language explanations of medical significance
- Risk factor identification and clinical recommendations
- Professional medical context for healthcare providers

**Robust Fallback System:**
- Primary/secondary AI provider redundancy
- Graceful degradation to ML-only mode
- Service health monitoring and status reporting

## CLI Application

The CLI application (`cli_app.py`) provides a powerful terminal interface with:

### CLI Features
- **Rich Terminal Interface**: Beautiful, color-coded terminal UI using Rich library
- **Interactive Conversation**: Natural language chat with AI assistant
- **File Processing**: Support for PDF, Word, TXT, JPEG, PNG files
- **Context Awareness**: Maintains conversation history and context
- **Medical Analysis**: Computer vision for images, NLP for documents
- **Service Integration**: Full access to all AI and RAG services

### CLI Commands
```bash
# Core Commands
predict interactive                    # Manual measurement input
predict file <path>                   # File-based prediction
file <path>                          # Process and analyze files
history                              # Show conversation history
history search <term>                # Search past interactions
history summary                      # Session statistics
status                               # Check AI service status
clear                                # Clear conversation history
help                                 # Show help information
quit                                 # Exit application
```

### CLI Workflow Examples
```bash
# Process medical document
file sample_medical_report.txt

# Analyze medical image
file mammogram.jpg
# Would you like AI-enhanced analysis? Y

# Interactive prediction with conversation context
predict interactive

# Ask contextual questions
What factors indicate malignancy in my recent prediction?

# Review session
history summary
```

### File Processing Capabilities
- **Medical Images**: Border analysis, texture analysis, shape characteristics
- **Medical Documents**: Entity extraction, measurement parsing, relevance assessment
- **Data Files**: Statistical analysis, medical column detection
- **Conversation Memory**: All processing results maintained in session context

### CLI Services Integration
- **Enhanced Prediction Service**: Full ML + AI + RAG pipeline
- **File Processor Service**: Multi-format file analysis with medical focus
- **Conversation Manager Service**: Context-aware history and session management
- **All AI Services**: Groq, HuggingFace, and RAG integration

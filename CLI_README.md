# 🩺 AI-Enhanced Breast Cancer Prediction CLI

A powerful command-line interface for the AI-Enhanced Breast Cancer Prediction System that supports interactive conversation, file processing, and context-aware analysis.

## ✨ Features

- **Interactive Conversation**: Natural language interaction with AI assistant
- **File Processing**: Support for PDF, Word, TXT, JPEG, PNG files
- **Context Awareness**: Maintains conversation history and context
- **AI-Enhanced Analysis**: Integration with Groq and HuggingFace AI services
- **Medical Image Analysis**: Computer vision analysis of medical images
- **Rich Terminal Interface**: Beautiful terminal UI with colors and formatting
- **Conversation History**: Search and review past interactions
- **Service Status Monitoring**: Real-time status of AI services

## 🚀 Quick Start

### 1. Launch the CLI

```bash
# Using the launcher script (recommended)
python run_cli.py

# Or directly
python cli_app.py

# With command line options
python cli_app.py --status              # Check service status
python cli_app.py --predict             # Start prediction mode
python cli_app.py --file report.pdf     # Process a specific file
```

### 2. Available Commands

Once in interactive mode, you can use these commands:

```bash
# Prediction Commands
predict interactive                      # Enter measurements manually
predict file sample_report.pdf          # Analyze file then predict

# File Processing
file mammogram.jpg                       # Process medical image
file patient_report.pdf                  # Process medical document
file data.csv                           # Process data file

# Conversation Management
history                                  # Show recent history
history search malignant                 # Search conversation
history summary                          # Session statistics

# System Commands
status                                   # Check AI service status
clear                                   # Clear conversation history
help                                    # Show help
quit                                    # Exit application
```

## 📁 Supported File Types

### 🖼️ Medical Images
- **Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **Analysis**: 
  - Border irregularity detection
  - Texture analysis
  - Shape characteristics
  - AI interpretation of findings

### 📄 Medical Documents
- **Formats**: `.pdf`, `.docx`, `.doc`, `.txt`
- **Analysis**:
  - Medical entity extraction
  - Measurement extraction
  - Relevance assessment
  - Document summarization

### 📊 Data Files
- **Formats**: `.csv`, `.xlsx`, `.json`
- **Analysis**:
  - Automatic medical column detection
  - Statistical summaries
  - Data validation

## 🔧 Setup Requirements

### Dependencies
Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with your API keys:

```bash
# AI Services
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Service Configuration
RAG_ENABLED=true
VECTOR_DB_PATH=./vector_db
MODEL_PATH=cancer_model.pkl
```

### Model File
Ensure the trained model file `cancer_model.pkl` is in the project directory.

## 💬 CLI Usage Examples

### Example 1: Interactive Prediction
```bash
📝 Enter command or ask a question: predict interactive

🔢 Enter tumor measurements

Radius Worst (Largest radius measurement) [16.0]: 18.5
Perimeter Worst (Largest perimeter measurement) [100.0]: 120.3
Area Worst (Largest area measurement) [800.0]: 1200.5
...

🎯 AI-Enhanced Prediction Result
⚠️ Prediction: Malignant
Confidence: 87.3%

🧠 AI Medical Analysis
Provider: Groq (llama3-8b-8192)

Based on the tumor measurements provided, this case shows several concerning features...
```

### Example 2: File Analysis
```bash
📝 Enter command or ask a question: file sample_medical_report.txt

📁 File Information
File: sample_medical_report.txt
Type: Document
Size: 0.01 MB

📄 Document Statistics:
• Words: 234
• Medical Relevance: high

🏥 Medical Entities Found:
• Diagnoses: malignant, carcinoma
• Procedures: mammography, biopsy
• Anatomy: breast, lymph node

📋 Document Summary
Medical imaging report showing suspicious breast mass with irregular characteristics...
```

### Example 3: Conversation History
```bash
📝 Enter command or ask a question: history search malignant

🔍 History Search - Search Results for 'malignant'

1. [01/15 14:30] high
Query: What does malignant mean?
Response: Malignant refers to cancerous tissue that can spread...

2. [01/15 14:25] medium  
Query: predict interactive
Response: Prediction: Malignant
```

## 🎯 Key Features in Detail

### Context-Aware Conversations
The CLI maintains conversation context across interactions:
- Remembers previously processed files
- References past predictions in new analyses
- Provides contextual AI responses based on session history

### Medical File Processing
Comprehensive analysis of medical files:
- **Images**: Computer vision analysis of tumor characteristics
- **Documents**: NLP extraction of medical entities and measurements  
- **Data**: Statistical analysis and medical column identification

### AI Integration
Multiple AI service integration:
- **Primary**: Groq API with Llama3-8B model
- **Fallback**: HuggingFace BioGPT model
- **RAG**: Retrieval-Augmented Generation for medical knowledge

### Rich Terminal Interface
Beautiful terminal experience:
- Color-coded output for different types of information
- Progress indicators for long-running operations
- Structured tables and panels for organized display
- Interactive prompts with validation

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Update Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. AI Services Unavailable**
```bash
# Check .env file exists and has valid keys
cat .env

# Test API connectivity
python -c "from services.groq_service import GroqService; print('✅ Groq OK')"
```

**3. Model File Missing**
```bash
# Train the model first
python train_cancer_model.py

# Or download pre-trained model
# (Model file should be named cancer_model.pkl)
```

**4. File Processing Errors**
- Ensure file paths are correct and accessible
- Check file format is supported
- Verify file size is under 50MB
- For images: ensure readable image format
- For documents: check file isn't corrupted

### Service Status Check
Use the status command to check all services:

```bash
📝 Enter command or ask a question: status

🔧 Service Status
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Service              ┃ Status      ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ ✅ Ml Model         │ Available   │
│ ✅ Groq Service     │ Available   │
│ ✅ Huggingface      │ Available   │
│ ✅ Rag Service      │ Available   │
└──────────────────────┴─────────────┘
```

## 🎉 Demo Workflow

Try this complete workflow to see all features:

1. **Start CLI**: `python run_cli.py`
2. **Check Status**: `status`
3. **Process Sample File**: `file sample_medical_report.txt`
4. **Make Prediction**: `predict interactive`
5. **Ask Questions**: `What factors indicate malignancy?`
6. **Review History**: `history`
7. **Search History**: `history search malignant`
8. **Get Summary**: `history summary`
9. **Exit**: `quit`

## 🔮 Advanced Usage

### Batch Processing
Process multiple files in sequence:

```bash
file report1.pdf
file image1.jpg  
file data1.csv
history summary  # See all processed files
```

### Medical Consultations
Use the AI assistant for medical questions:

```bash
📝 What are the key features that distinguish malignant from benign tumors?
📝 How does tumor size correlate with malignancy risk?
📝 What should I look for in mammography reports?
```

### Image Analysis Workflow
For medical images:

```bash
file mammogram.jpg                    # Process image
🧠 Would you like AI-enhanced analysis? Y   # Get AI interpretation
predict interactive                   # Follow up with prediction
history search image                  # Review image analyses
```

This CLI provides a comprehensive, user-friendly interface to leverage the full power of the AI-Enhanced Breast Cancer Prediction System through the terminal.

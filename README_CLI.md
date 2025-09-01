# AI-Enhanced Breast Cancer Screening Assistant

## 🎯 **FULLY CONSOLIDATED DUAL-MODE APPLICATION**

Everything has been merged into a single file: **`main.py`**

## 🚀 **Quick Start**

### **CLI Screening Mode (Default)**
```bash
python main.py
```

### **Web API Server Mode**
```bash
python main.py --api
```

### **Help & Options**
```bash
python main.py --help
```

## 📋 **Complete Consolidation**

**Removed Files:**
- ✅ `cli_app.py` - Merged into main.py
- ✅ `demo_cli.py` - Functionality integrated  
- ✅ `medical_screening_cli.py` - Merged into main.py
- ✅ `run_cli.py` - No longer needed
- ✅ `screening_cli.py` - Merged into main.py
- ✅ `api_server.py` - Merged into main.py
- ✅ `start.py` - No longer needed

**Single Consolidated File:**
- ✅ `main.py` - **Complete dual-mode application (CLI + API)**

## 🩺 **Application Features**

### **Immediate Screening (No Menus)**
- Starts directly with 11 evidence-based questions
- Real-time accuracy assessment for each response
- Medical file upload during screening
- Risk calculation with evidence-based weighting

### **Comprehensive Analysis**
- AI-powered medical interpretation
- Risk level determination (Low/Moderate/High)
- Category-based score breakdown
- Professional recommendations

### **Interactive Conversation**
- Context-aware medical Q&A
- File upload and analysis
- Screening results review
- Medical education and support

### **Session Management**
- Conversation history tracking
- Session duration monitoring
- Comprehensive exit summary
- Logging for debugging

## 🔧 **System Requirements**

**Python Packages:**
```bash
pip install rich loguru python-dotenv asyncio pathlib json datetime
```

**Service Dependencies:**
- `services/enhanced_prediction_service.py`
- `services/file_processor.py`
- `services/conversation_manager.py`

## 🎯 **Usage Examples**

### **Basic Screening**
```bash
python main.py
# Automatically starts 11-question screening
# Provides AI analysis and recommendations
# Enables follow-up conversation
```

### **With File Upload**
- Answer "Yes" to question 11 about medical file upload
- Provide full path to medical documents (PDF, DOC, images)
- Get integrated analysis combining screening + file

### **Conversation Commands**
After screening completion:
- Ask any medical question naturally
- Type `upload` or `file` to add medical documents
- Type `summary` or `results` to review screening
- Type `quit` or `exit` to end with session summary

## 🌐 **Web API Mode**

To run the FastAPI web server:
```bash
python main.py --api
# Runs on http://localhost:8000
# Interactive docs: http://localhost:8000/docs
# Health check: http://localhost:8000/health
# Status: http://localhost:8000/status
```

**API Endpoints:**
- `POST /predict` - Basic ML prediction
- `POST /predict/enhanced` - AI-enhanced prediction with RAG
- `GET /status` - Service status and capabilities
- `GET /health` - Detailed health check
- `GET /` - API information

## 📊 **Question Categories**

1. **Physical Symptoms** (5 questions)
   - Lumps/thickening, breast changes, nipple discharge, skin changes, pain

2. **Genetic History** (1 question)  
   - Family/personal cancer history

3. **Hormonal Factors** (2 questions)
   - Menstrual/menopause history, pregnancy/HRT status

4. **Medical History** (1 question)
   - Previous procedures and tests

5. **General Health** (1 question)
   - Unexplained symptoms

6. **Supplementary** (1 question)
   - Medical file upload option

## ⚠️ **Important Notes**

- **Educational tool only** - Not for medical diagnosis
- **Immediate medical attention** for concerning symptoms
- **Professional consultation** always recommended
- **Evidence-based** risk assessment algorithm
- **Privacy-focused** - All data processed locally

## 🆘 **Support**

If you encounter issues:
1. Check that all service files are present in `services/` directory
2. Verify environment variables in `.env` file
3. Review logs in `logs/` directory for debugging
4. Ensure all Python packages are installed

---

**🩺 Your health matters - this tool helps you stay informed and proactive about breast cancer screening!**

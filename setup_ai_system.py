"""
Setup script for initializing the AI-enhanced breast cancer prediction system
Run this after installing dependencies to verify everything works correctly
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if all required environment variables are set"""
    load_dotenv()
    
    required_vars = ["GROQ_API_KEY", "HUGGINGFACE_API_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_model_file():
    """Check if the ML model file exists"""
    model_path = Path("cancer_model.pkl")
    if model_path.exists():
        print(f"✅ ML model found: {model_path}")
        return True
    else:
        print(f"❌ ML model not found: {model_path}")
        print("Please ensure cancer_model.pkl is in the project root")
        return False

def check_knowledge_base():
    """Check if medical knowledge base exists"""
    knowledge_path = Path("rag_data/medical_knowledge.txt")
    if knowledge_path.exists():
        print(f"✅ Medical knowledge base found: {knowledge_path}")
        return True
    else:
        print(f"❌ Medical knowledge base not found: {knowledge_path}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("groq", "Groq API"),
        ("transformers", "HuggingFace Transformers"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("requests", "Requests"),
        ("dotenv", "python-dotenv"),
        ("loguru", "Loguru")
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are available")
    return True

def test_services():
    """Test if AI services can be initialized"""
    try:
        print("\n🧪 Testing AI services initialization...")
        
        # Test imports
        from services.enhanced_prediction_service import EnhancedPredictionService
        print("✅ Enhanced prediction service import successful")
        
        # Initialize services (this will show which ones work)
        service = EnhancedPredictionService()
        status = service._get_services_status()
        
        print("\n📊 Service Status:")
        for service_name, available in status.items():
            icon = "✅" if available else "❌"
            print(f"{icon} {service_name.replace('_', ' ').title()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing services: {str(e)}")
        return False

def main():
    """Main setup verification"""
    print("🚀 AI-Enhanced Breast Cancer Prediction System Setup")
    print("=" * 60)
    
    checks = [
        ("Environment Variables", check_environment),
        ("ML Model File", check_model_file),
        ("Medical Knowledge Base", check_knowledge_base),
        ("Package Imports", test_imports),
        ("AI Services", test_services)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Setup verification completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start the FastAPI server: uvicorn main:app --reload")
        print("2. Start the Streamlit UI: streamlit run app_ui.py")
        print("3. Open http://127.0.0.1:8000/docs for API documentation")
    else:
        print("❌ Setup verification failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

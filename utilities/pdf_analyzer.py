"""
PDF Analyzer Tool - Helps analyze and extract readable content from the benchmark PDF report
"""

import sys
import os
from pathlib import Path

def analyze_pdf_with_basic_tools():
    """Analyze PDF using available Python libraries"""
    
    # Try to find the latest PDF report (including improved versions)
    pdf_files = list(Path('.').glob('*breast_cancer_benchmark_report_*.pdf'))
    if not pdf_files:
        print("❌ No benchmark PDF reports found")
        return
    
    # Get the latest PDF
    latest_pdf = max(pdf_files, key=os.path.getctime)
    print(f"📄 Analyzing PDF: {latest_pdf}")
    
    # Try different PDF reading approaches
    methods_tried = []
    
    # Method 1: PyPDF2
    try:
        import PyPDF2
        print("\n🔍 Trying PyPDF2...")
        with open(latest_pdf, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"✅ PDF has {len(pdf_reader.pages)} pages")
            
            # Extract text from first few pages
            for i in range(min(3, len(pdf_reader.pages))):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                print(f"\n📖 Page {i+1} content preview:")
                print("=" * 40)
                print(text[:500] if text else "No extractable text found")
                print("=" * 40)
        methods_tried.append("PyPDF2 - Success")
        
    except ImportError:
        print("⚠️  PyPDF2 not available")
        methods_tried.append("PyPDF2 - Not installed")
    except Exception as e:
        print(f"❌ PyPDF2 failed: {str(e)}")
        methods_tried.append(f"PyPDF2 - Error: {str(e)}")
    
    # Method 2: pdfplumber
    try:
        import pdfplumber
        print("\n🔍 Trying pdfplumber...")
        with pdfplumber.open(latest_pdf) as pdf:
            print(f"✅ PDF has {len(pdf.pages)} pages")
            
            for i in range(min(3, len(pdf.pages))):
                page = pdf.pages[i]
                text = page.extract_text()
                print(f"\n📖 Page {i+1} content (pdfplumber):")
                print("=" * 40)
                print(text[:500] if text else "No extractable text found")
                print("=" * 40)
        methods_tried.append("pdfplumber - Success")
        
    except ImportError:
        print("⚠️  pdfplumber not available")
        methods_tried.append("pdfplumber - Not installed")
    except Exception as e:
        print(f"❌ pdfplumber failed: {str(e)}")
        methods_tried.append(f"pdfplumber - Error: {str(e)}")
    
    # Method 3: Check file properties
    try:
        file_size = os.path.getsize(latest_pdf)
        print(f"\n📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        if file_size < 1000:
            print("⚠️  File seems very small - might be corrupted or empty")
        elif file_size > 10*1024*1024:
            print("⚠️  File seems very large - might have embedded issues")
        else:
            print("✅ File size seems normal")
            
    except Exception as e:
        print(f"❌ Could not check file properties: {str(e)}")
    
    # Summary
    print(f"\n📋 Analysis Summary:")
    for method in methods_tried:
        print(f"   • {method}")
    
    return latest_pdf

def suggest_fixes():
    """Suggest potential fixes for PDF readability issues"""
    print(f"\n💡 Potential Issues and Solutions:")
    print("=" * 50)
    
    issues_and_fixes = [
        {
            "issue": "Text not rendering properly",
            "solutions": [
                "Font embedding issues in ReportLab",
                "Use different PDF viewer (Adobe Reader vs browser)",
                "Check if special characters are causing problems",
                "Verify font paths and availability"
            ]
        },
        {
            "issue": "Charts/Images not displaying",
            "solutions": [
                "Check if PNG files exist in benchmark_charts folder",
                "Verify image paths in PDF generation code",
                "Ensure proper image embedding in ReportLab",
                "Check image file permissions"
            ]
        },
        {
            "issue": "Layout/Formatting problems",
            "solutions": [
                "Table styling issues in ReportLab",
                "Page break problems",
                "Spacing and margin issues",
                "Custom style definitions not working"
            ]
        },
        {
            "issue": "PDF corruption",
            "solutions": [
                "ReportLab build process interrupted",
                "File writing permissions",
                "Memory issues during generation",
                "Antivirus interfering with file creation"
            ]
        }
    ]
    
    for idx, problem in enumerate(issues_and_fixes, 1):
        print(f"{idx}. {problem['issue']}:")
        for solution in problem['solutions']:
            print(f"   • {solution}")
        print()

def check_dependencies():
    """Check what PDF libraries are available"""
    print("🔍 Checking PDF processing capabilities...")
    
    libraries = [
        ("PyPDF2", "pip install PyPDF2"),
        ("pdfplumber", "pip install pdfplumber"), 
        ("reportlab", "pip install reportlab"),
        ("matplotlib", "pip install matplotlib"),
        ("seaborn", "pip install seaborn")
    ]
    
    for lib_name, install_cmd in libraries:
        try:
            __import__(lib_name.lower() if lib_name != "PyPDF2" else "PyPDF2")
            print(f"✅ {lib_name} is available")
        except ImportError:
            print(f"❌ {lib_name} not found - install with: {install_cmd}")

def main():
    print("🔍 PDF Report Analyzer")
    print("=" * 40)
    
    # Check dependencies first
    check_dependencies()
    print()
    
    # Analyze the PDF
    pdf_file = analyze_pdf_with_basic_tools()
    
    # Suggest fixes
    suggest_fixes()
    
    if pdf_file:
        print(f"\n🎯 Recommendations:")
        print(f"1. Try opening {pdf_file} with different PDF viewers")
        print("2. Check the benchmark_charts folder for image files")
        print("3. Re-run the benchmark if the PDF seems corrupted")
        print("4. Install missing PDF libraries for better analysis")

if __name__ == "__main__":
    main()

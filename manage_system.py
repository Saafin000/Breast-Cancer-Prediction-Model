#!/usr/bin/env python3
"""
System Management Script for Breast Cancer Screening Tool
Provides easy commands to run, test, and maintain the consolidated system
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

class SystemManager:
    def __init__(self):
        self.scripts = {
            'benchmark': 'benchmark_main.py',
            'train': 'model_training_main.py', 
            'main': 'main.py',
            'analyze_pdf': 'utilities/pdf_analyzer.py'
        }
        
    def show_status(self):
        """Show current system status"""
        print("🔍 Breast Cancer Screening System Status")
        print("=" * 50)
        
        # Check main files
        print("📁 Main Files:")
        for name, script in self.scripts.items():
            if os.path.exists(script):
                size = os.path.getsize(script)
                modified = datetime.fromtimestamp(os.path.getmtime(script))
                print(f"   ✅ {script} ({size:,} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"   ❌ {script} - Missing")
        
        # Check model files
        print("\n🤖 Model Files:")
        model_files = ['cancer_model.pkl', 'model_scaler.pkl', 'model_selector.pkl']
        for model_file in model_files:
            if os.path.exists(model_file):
                size = os.path.getsize(model_file)
                modified = datetime.fromtimestamp(os.path.getmtime(model_file))
                print(f"   ✅ {model_file} ({size:,} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"   ❌ {model_file} - Missing")
        
        # Check backup directory
        print("\n💾 Backup Files:")
        if os.path.exists('backup_scripts'):
            backup_files = list(Path('backup_scripts').glob('*.py'))
            print(f"   📁 {len(backup_files)} files in backup_scripts/")
            for f in sorted(backup_files):
                print(f"      • {f.name}")
        else:
            print("   ❌ backup_scripts directory not found")
        
        # Check utilities
        print("\n🔧 Utilities:")
        if os.path.exists('utilities'):
            util_files = list(Path('utilities').glob('*.py'))
            print(f"   📁 {len(util_files)} files in utilities/")
            for f in sorted(util_files):
                print(f"      • {f.name}")
        else:
            print("   ❌ utilities directory not found")
        
        # Check recent reports
        print("\n📄 Recent Reports:")
        pdf_files = list(Path('.').glob('*.pdf'))
        if pdf_files:
            recent_pdfs = sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
            for pdf in recent_pdfs:
                modified = datetime.fromtimestamp(pdf.stat().st_mtime)
                size = pdf.stat().st_size
                print(f"   📄 {pdf.name} ({size:,} bytes, {modified.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("   ❌ No PDF reports found")
            
    def run_benchmark(self):
        """Run the benchmark system"""
        print("🧬 Running Breast Cancer Benchmark...")
        try:
            subprocess.run([sys.executable, self.scripts['benchmark']], check=True)
            print("✅ Benchmark completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Benchmark failed: {e}")
        except Exception as e:
            print(f"❌ Error running benchmark: {e}")
    
    def retrain_model(self):
        """Retrain the model for improved accuracy"""
        print("🔄 Retraining Model...")
        try:
            subprocess.run([sys.executable, self.scripts['train']], check=True)
            print("✅ Model retraining completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Model training failed: {e}")
        except Exception as e:
            print(f"❌ Error training model: {e}")
    
    def start_main_app(self):
        """Start the main application"""
        print("🚀 Starting Main Application...")
        try:
            subprocess.run([sys.executable, self.scripts['main']], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Main app failed: {e}")
        except Exception as e:
            print(f"❌ Error starting main app: {e}")
    
    def analyze_pdfs(self):
        """Analyze PDF reports for readability issues"""
        print("🔍 Analyzing PDF Reports...")
        try:
            subprocess.run([sys.executable, self.scripts['analyze_pdf']], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ PDF analysis failed: {e}")
        except Exception as e:
            print(f"❌ Error analyzing PDFs: {e}")
    
    def cleanup_old_reports(self):
        """Clean up old PDF reports (keep last 5)"""
        print("🧹 Cleaning up old reports...")
        pdf_files = list(Path('.').glob('*.pdf'))
        if len(pdf_files) > 5:
            # Sort by modification time, keep newest 5
            sorted_pdfs = sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)
            to_delete = sorted_pdfs[5:]
            
            for pdf in to_delete:
                try:
                    pdf.unlink()
                    print(f"   🗑️ Deleted: {pdf.name}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {pdf.name}: {e}")
            
            print(f"✅ Cleanup completed, kept {len(sorted_pdfs[:5])} newest reports")
        else:
            print(f"✅ No cleanup needed, {len(pdf_files)} reports found")

def show_help():
    """Show available commands"""
    print("🔧 Breast Cancer Screening System Manager")
    print("=" * 50)
    print("Available commands:")
    print("  status     - Show system status and file overview")
    print("  benchmark  - Run comprehensive benchmark")
    print("  train      - Retrain model for improved accuracy")
    print("  run        - Start the main application")
    print("  analyze    - Analyze PDF reports for issues")
    print("  cleanup    - Clean up old PDF reports")
    print("  help       - Show this help message")
    print("\nUsage: python manage_system.py <command>")
    print("\nExamples:")
    print("  python manage_system.py status")
    print("  python manage_system.py benchmark")
    print("  python manage_system.py train")

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    manager = SystemManager()
    
    if command == 'status':
        manager.show_status()
    elif command == 'benchmark':
        manager.run_benchmark()
    elif command == 'train':
        manager.retrain_model()
    elif command == 'run':
        manager.start_main_app()
    elif command == 'analyze':
        manager.analyze_pdfs()
    elif command == 'cleanup':
        manager.cleanup_old_reports()
    elif command == 'help':
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python manage_system.py help' for available commands")

if __name__ == "__main__":
    main()

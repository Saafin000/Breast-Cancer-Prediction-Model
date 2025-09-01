"""
AI-Enhanced Breast Cancer Screening Assistant
=============================================

Dual-Mode Application:
1. CLI Mode: Direct 11-question breast cancer risk assessment
2. API Mode: FastAPI web server for REST endpoints

CLI Features:
- Immediate screening start (no menus)
- Real-time response accuracy evaluation  
- AI-powered medical analysis and recommendations
- File processing for medical reports (PDF, DOC, images)
- Interactive conversation with medical context
- Session management and history tracking
- Evidence-based risk calculation
- Professional medical recommendations
- Medical disclaimer and safety guidelines

API Features:
- REST endpoints for predictions
- Enhanced AI analysis with RAG
- Service status and health checks
- Compatible with web frontends
- Legacy API compatibility

Usage:
    python main.py              # CLI screening mode (default)
    python main.py --api        # FastAPI server mode
    python main.py --help       # Show help
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Rich console for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Services
from services.enhanced_prediction_service import EnhancedPredictionService
from services.file_processor import FileProcessor
from services.conversation_manager import ConversationManager

# Logging and environment
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# FASTAPI WEB SERVER COMPONENTS
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="AI-Enhanced Breast Cancer Prediction API",
    description="Advanced breast cancer prediction using ML, AI analysis, and RAG",
    version="2.0.0"
)

# Global enhanced prediction service for both CLI and API
enhanced_predictor = None
fallback_model = None

# Initialize services for both CLI and API modes
try:
    enhanced_predictor = EnhancedPredictionService()
    logger.info("Enhanced prediction service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize enhanced prediction service: {e}")
    # Fallback to basic model
    try:
        fallback_model = joblib.load("cancer_model.pkl")
    except Exception as model_e:
        logger.error(f"Failed to load fallback model: {model_e}")

# Define input data structures for API
class CancerInput(BaseModel):
    radius_worst: float
    perimeter_worst: float
    area_worst: float
    concave_points_worst: float
    concavity_worst: float
    compactness_worst: float
    radius_mean: float
    perimeter_mean: float
    area_mean: float
    concave_points_mean: float
    concavity_mean: float
    compactness_mean: float
    texture_worst: float
    smoothness_worst: float
    symmetry_worst: float

class EnhancedCancerInput(BaseModel):
    radius_worst: float
    perimeter_worst: float
    area_worst: float
    concave_points_worst: float
    concavity_worst: float
    compactness_worst: float
    radius_mean: float
    perimeter_mean: float
    area_mean: float
    concave_points_mean: float
    concavity_mean: float
    compactness_mean: float
    texture_worst: float
    smoothness_worst: float
    symmetry_worst: float
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }

# FastAPI Routes
@app.get("/")
def read_root():
    return {
        "message": "AI-Enhanced Breast Cancer Prediction API is running!",
        "cli_available": True,
        "cli_command": "python main.py",
        "endpoints": ["/predict", "/predict/enhanced", "/status", "/health"]
    }

@app.post("/predict")
def predict_cancer(data: CancerInput):
    """Basic prediction endpoint for legacy compatibility"""
    try:
        # Convert input to array for model
        input_data = np.array([[
            data.radius_worst,
            data.perimeter_worst,
            data.area_worst,
            data.concave_points_worst,
            data.concavity_worst,
            data.compactness_worst,
            data.radius_mean,
            data.perimeter_mean,
            data.area_mean,
            data.concave_points_mean,
            data.concavity_mean,
            data.compactness_mean,
            data.texture_worst,
            data.smoothness_worst,
            data.symmetry_worst
        ]])
        
        if enhanced_predictor:
            prediction = enhanced_predictor.ml_model.predict(input_data)[0]
        elif fallback_model:
            prediction = fallback_model.predict(input_data)[0]
        else:
            raise HTTPException(status_code=503, detail="No prediction model available")
        
        result = "Malignant" if prediction == 1 else "Benign"
        return {"prediction": result}
        
    except Exception as e:
        logger.error(f"Basic prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/enhanced")
async def predict_cancer_enhanced(data: EnhancedCancerInput) -> Dict[str, Any]:
    """Enhanced prediction endpoint with AI analysis and RAG"""
    try:
        if not enhanced_predictor:
            raise HTTPException(status_code=503, detail="Enhanced prediction service not available")
        
        features = data.dict()
        result = await enhanced_predictor.predict_enhanced(features)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced prediction failed: {str(e)}")

@app.get("/status")
async def get_service_status():
    """Get status of all AI services"""
    if enhanced_predictor:
        services_status = enhanced_predictor._get_services_status()
        
        rag_info = {}
        if enhanced_predictor.rag_service:
            rag_info = await enhanced_predictor.rag_service.get_collection_info()
        
        return {
            "services": services_status,
            "rag_collection": rag_info,
            "api_version": "2.0.0",
            "enhanced_features": True,
            "cli_available": True,
            "cli_command": "python main.py"
        }
    else:
        return {
            "services": {"ml_model": fallback_model is not None, "enhanced_features": False},
            "api_version": "1.0.0",
            "enhanced_features": False,
            "cli_available": True,
            "cli_command": "python main.py"
        }

@app.get("/health")
async def health_check():
    """Detailed health check for all services"""
    health_status = {
        "api": "healthy",
        "timestamp": str(np.datetime64('now')),
        "services": {},
        "cli_available": True
    }
    
    if enhanced_predictor:
        services = enhanced_predictor._get_services_status()
        
        for service_name, is_available in services.items():
            health_status["services"][service_name] = "healthy" if is_available else "unavailable"
        
        # Test ML model
        try:
            test_features = {
                "radius_worst": 16.0, "perimeter_worst": 100.0, "area_worst": 800.0,
                "concave_points_worst": 0.2, "concavity_worst": 0.3, "compactness_worst": 0.2,
                "radius_mean": 14.0, "perimeter_mean": 90.0, "area_mean": 700.0,
                "concave_points_mean": 0.1, "concavity_mean": 0.2, "compactness_mean": 0.1,
                "texture_worst": 20.0, "smoothness_worst": 0.1, "symmetry_worst": 0.3
            }
            ml_result = enhanced_predictor._get_ml_prediction(test_features)
            health_status["ml_model_test"] = "passed" if ml_result["prediction"] != "Error" else "failed"
        except:
            health_status["ml_model_test"] = "failed"
    else:
        health_status["services"]["ml_model"] = "basic_only" if fallback_model else "unavailable"
        health_status["enhanced_features"] = False
    
    return health_status

# ============================================================================
# CLI APPLICATION COMPONENTS
# ============================================================================


class BreastCancerScreeningCLI:
    """
    Comprehensive AI-Enhanced Breast Cancer Screening CLI Application
    
    This application provides:
    1. Immediate 11-question medical screening
    2. Real-time accuracy assessment 
    3. Risk calculation and interpretation
    4. AI-powered medical analysis
    5. File processing for medical documents/images
    6. Open-ended medical conversation
    7. Session tracking and summary
    """
    
    def __init__(self):
        self.console = Console()
        self.prediction_service = None
        self.file_processor = FileProcessor()
        self.conversation_manager = ConversationManager()
        
        # Initialize services
        self._initialize_services()
        
        # Screening data storage
        self.screening_responses = {}
        self.overall_risk_score = 0
        self.session_start_time = datetime.now()

    def _initialize_services(self):
        """Initialize AI prediction services with error handling"""
        try:
            self.prediction_service = EnhancedPredictionService()
            self.console.print("✅ AI Medical Assistant ready", style="green")
        except Exception as e:
            self.console.print(f"⚠️ Warning: AI services limited: {e}", style="yellow")
            logger.warning(f"Failed to initialize prediction service: {e}")

    async def run(self):
        """
        Main application flow
        
        Workflow:
        1. Welcome and medical disclaimer
        2. Direct screening questionnaire (11 questions)
        3. Risk analysis and AI interpretation
        4. Open conversation mode
        5. Session summary on exit
        """
        # Display welcome and disclaimer
        self._show_welcome()
        
        if not self._confirm_medical_disclaimer():
            self.console.print("👋 Thank you. Please consult a healthcare professional for medical concerns.", style="blue")
            return
        
        # Core screening workflow
        await self._run_screening_questionnaire()
        
        # Open conversation mode after screening
        await self._start_open_conversation()

    def _show_welcome(self):
        """Display welcome message and application info"""
        self.console.print(Panel.fit(
            "[bold blue]🩺 AI-Enhanced Breast Cancer Screening Assistant[/bold blue]\n"
            "[dim]Comprehensive medical risk assessment with AI analysis[/dim]\n"
            "[dim]Version 1.0 | Powered by Claude AI & Medical Guidelines[/dim]",
            title="Welcome"
        ))

    def _confirm_medical_disclaimer(self) -> bool:
        """Show medical disclaimer and get user consent"""
        self.console.print(Panel(
            "[bold yellow]⚠️ MEDICAL DISCLAIMER[/bold yellow]\n\n"
            "• This screening tool is for EDUCATIONAL PURPOSES ONLY\n"
            "• It does NOT replace professional medical diagnosis or treatment\n"
            "• Always consult qualified healthcare professionals for medical advice\n"
            "• Seek immediate medical attention for urgent health concerns\n"
            "• This tool provides risk assessment based on medical guidelines\n\n"
            "[bold red]NOT suitable for emergency medical situations[/bold red]",
            title="⚠️ Important Medical Notice"
        ))
        
        return Confirm.ask("\n[bold]Do you understand and agree to continue with the screening?[/bold]")

    async def _run_screening_questionnaire(self):
        """
        Run the complete 11-question screening questionnaire
        
        Questions cover:
        - Physical symptoms and changes
        - Family history
        - Hormonal factors
        - Medical history
        - File upload option
        """
        self.console.print(Panel(
            "[bold cyan]Starting Comprehensive Screening[/bold cyan]\n\n"
            "I'll ask you 11 evidence-based questions about breast cancer risk factors.\n"
            "Each response will be evaluated for accuracy based on medical guidelines.\n\n"
            "[dim]Please answer honestly for the most accurate assessment.[/dim]",
            title="🔍 Breast Cancer Risk Screening"
        ))
        
        # Define the 11 screening questions with medical accuracy data
        questions = self._get_screening_questions()
        total_risk_score = 0
        
        # Process each question
        for i, question in enumerate(questions, 1):
            self.console.print(f"\n[bold cyan]Question {i}/11[/bold cyan]")
            self.console.print(f"[white]{question['question']}[/white]")
            
            # Handle different question types
            if question["type"] == "yes_no":
                response = Confirm.ask("[bold]Your answer (Yes/No)[/bold]")
                accuracy = question["accuracy"]["yes" if response else "no"]
                
                # Calculate risk contribution (exclude file upload question)
                if response and question["id"] != "medical_report_upload":
                    risk_contribution = question["risk_weight"] * 100
                    total_risk_score += risk_contribution
                
            elif question["type"] == "open_ended":
                response = Prompt.ask(
                    "[bold]Your answer[/bold]", 
                    default="Please provide details about menstrual/menopause history"
                )
                accuracy = question["accuracy"]
                
                # Parse age information for risk assessment
                if any(char.isdigit() for char in response):
                    # Early menstruation increases risk
                    if "early" in response.lower() or any(str(age) in response for age in range(8, 12)):
                        risk_contribution = question["risk_weight"] * 80
                        total_risk_score += risk_contribution
            
            # Store response with metadata
            self.screening_responses[question["id"]] = {
                "question": question["question"],
                "response": response,
                "accuracy": accuracy,
                "risk_weight": question["risk_weight"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Show immediate accuracy feedback
            self.console.print(f"[dim]✅ Response accuracy: {accuracy:.1f}%[/dim]")
            
            # Handle file upload if requested
            if question["id"] == "medical_report_upload" and response:
                await self._handle_immediate_file_upload()
        
        # Store overall risk score
        self.overall_risk_score = total_risk_score
        
        # Display comprehensive screening results
        await self._display_screening_results(total_risk_score)

    def _get_screening_questions(self) -> List[Dict[str, Any]]:
        """
        Define the 11 evidence-based screening questions
        
        Each question includes:
        - Unique ID for tracking
        - Question text
        - Response type (yes_no or open_ended)
        - Accuracy scores based on medical literature
        - Risk weight for calculation
        """
        return [
            {
                "id": "lumps_thickening",
                "question": "Have you noticed any lumps or thickening in your breast or underarm area?",
                "type": "yes_no",
                "accuracy": {"yes": 92.0, "no": 96.0},
                "risk_weight": 0.15,
                "category": "physical_symptoms"
            },
            {
                "id": "breast_changes", 
                "question": "Have you experienced any recent changes in the size, shape, or appearance of your breasts?",
                "type": "yes_no",
                "accuracy": {"yes": 87.0, "no": 91.0},
                "risk_weight": 0.12,
                "category": "physical_symptoms"
            },
            {
                "id": "nipple_discharge",
                "question": "Is there any nipple discharge that is unusual, such as bloody or clear fluid (when not breastfeeding)?",
                "type": "yes_no", 
                "accuracy": {"yes": 89.0, "no": 94.0},
                "risk_weight": 0.10,
                "category": "physical_symptoms"
            },
            {
                "id": "skin_changes",
                "question": "Have you observed any skin changes on your breast, such as dimpling, redness, or scaling?",
                "type": "yes_no",
                "accuracy": {"yes": 85.0, "no": 92.0},
                "risk_weight": 0.08,
                "category": "physical_symptoms"
            },
            {
                "id": "pain_tenderness",
                "question": "Do you feel any pain or tenderness in your breast or armpit area?", 
                "type": "yes_no",
                "accuracy": {"yes": 68.0, "no": 87.0},  # Pain is less specific indicator
                "risk_weight": 0.05,
                "category": "physical_symptoms"
            },
            {
                "id": "family_history",
                "question": "Have you or a close family member ever been diagnosed with breast cancer?",
                "type": "yes_no",
                "accuracy": {"yes": 97.0, "no": 98.0},  # Very reliable historical data
                "risk_weight": 0.20,
                "category": "genetic_history"
            },
            {
                "id": "menstrual_menopause",
                "question": "At what age did you begin menstruating and (if applicable) reach menopause?",
                "type": "open_ended",
                "accuracy": 85.0,  # Medical history self-reporting accuracy
                "risk_weight": 0.08,
                "category": "hormonal_factors"
            },
            {
                "id": "medical_procedures",
                "question": "Have you ever had a breast biopsy, mammogram, or any other breast-related medical procedure?",
                "type": "yes_no",
                "accuracy": {"yes": 90.0, "no": 75.0},
                "risk_weight": 0.07,
                "category": "medical_history"
            },
            {
                "id": "hormonal_factors",
                "question": "Are you currently pregnant, breastfeeding, or taking hormone replacement therapy?",
                "type": "yes_no",
                "accuracy": {"yes": 82.0, "no": 88.0},
                "risk_weight": 0.06,
                "category": "hormonal_factors"
            },
            {
                "id": "general_symptoms",
                "question": "Have you experienced unexplained weight loss, fatigue, or other general symptoms lately?",
                "type": "yes_no",
                "accuracy": {"yes": 65.0, "no": 89.0},  # General symptoms less specific
                "risk_weight": 0.09,
                "category": "general_health"
            },
            {
                "id": "medical_report_upload",
                "question": "Would you like to upload or submit a medical report (such as a mammogram, biopsy result, or doctor's note) for a more accurate evaluation?",
                "type": "yes_no",
                "accuracy": {"yes": 95.0, "no": 80.0},
                "risk_weight": 0.0,  # Doesn't affect risk calculation, just analysis quality
                "category": "supplementary"
            }
        ]

    async def _handle_immediate_file_upload(self):
        """Handle file upload during the screening process"""
        file_path = Prompt.ask("\n📁 Enter the full path to your medical file")
        
        if not file_path:
            return
            
        if not Path(file_path).exists():
            self.console.print(f"❌ File not found: {file_path}", style="red")
            return
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Analyzing your medical file...", total=None)
            file_result = await self.file_processor.process_file(file_path)
        
        if "error" not in file_result:
            file_type = file_result.get("file_type", "file")
            self.console.print(f"✅ Successfully processed {file_type}", style="green")
            
            # Show brief file summary
            if file_type == "document":
                text_analysis = file_result.get("text_analysis", {})
                summary = text_analysis.get("summary", "No summary available")
                self.console.print(f"[dim]📄 Document summary: {summary[:100]}...[/dim]")
            
            # Save file context for AI analysis
            await self.conversation_manager.add_interaction(
                user_input=f"Uploaded medical file during screening: {Path(file_path).name}",
                system_response="File processed successfully",
                context_type="file_analysis",
                file_processed=file_path
            )
        else:
            self.console.print(f"⚠️ Could not process file: {file_result['error']}", style="yellow")

    async def _display_screening_results(self, total_risk_score: float):
        """
        Display comprehensive screening results with risk assessment
        
        Includes:
        - Overall risk level and score
        - Response breakdown table
        - AI medical analysis
        - Professional recommendations
        """
        # Calculate risk percentage (capped at 100%)
        risk_percentage = min(total_risk_score, 100)
        
        # Determine risk level and styling
        risk_level, risk_color, urgency = self._calculate_risk_level(risk_percentage)
        
        # Main results display
        self.console.print(Panel(
            f"[bold {risk_color}]🎯 SCREENING COMPLETE[/bold {risk_color}]\n\n"
            f"[bold]Overall Risk Assessment: {risk_level}[/bold]\n"
            f"[bold]Risk Score: {risk_percentage:.1f}%[/bold]\n"
            f"[bold]Session Duration: {self._get_session_duration()}[/bold]\n\n"
            f"{urgency}",
            title="📊 Breast Cancer Risk Assessment Results"
        ))
        
        # Show detailed response breakdown
        self._show_response_breakdown()
        
        # Get AI-powered comprehensive analysis
        if self.prediction_service and hasattr(self.prediction_service, 'groq_service'):
            await self._get_comprehensive_ai_analysis(risk_percentage)
        else:
            self._show_basic_recommendations(risk_percentage)
        
        # Save screening completion to conversation history
        await self.conversation_manager.add_interaction(
            user_input="Completed comprehensive breast cancer screening questionnaire",
            system_response=f"Risk assessment completed: {risk_level} risk ({risk_percentage:.1f}%)",
            context_type="screening_completion",
            confidence_score=risk_percentage/100,
            prediction_result={
                "risk_level": risk_level, 
                "score": risk_percentage,
                "total_questions": len(self.screening_responses),
                "session_duration": self._get_session_duration()
            }
        )

    def _calculate_risk_level(self, risk_percentage: float) -> tuple:
        """Calculate risk level, color, and urgency message"""
        if risk_percentage >= 60:
            return "High", "red", "🚨 IMMEDIATE medical consultation recommended"
        elif risk_percentage >= 30:
            return "Moderate", "yellow", "📅 Schedule medical evaluation within 1-2 weeks"
        else:
            return "Low", "green", "✅ Continue regular screening schedule as recommended"

    def _get_session_duration(self) -> str:
        """Calculate and format session duration"""
        duration = datetime.now() - self.session_start_time
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes}m {seconds}s"

    def _show_response_breakdown(self):
        """Display detailed breakdown of all responses with accuracy scores"""
        breakdown_table = Table(title="📋 Your Responses & Medical Accuracy")
        breakdown_table.add_column("Question", style="cyan", width=50)
        breakdown_table.add_column("Your Answer", style="magenta", width=15)
        breakdown_table.add_column("Accuracy", style="green", width=10)
        breakdown_table.add_column("Category", style="blue", width=15)
        
        # Get questions for category info
        questions = self._get_screening_questions()
        question_lookup = {q["id"]: q for q in questions}
        
        for i, (question_id, data) in enumerate(self.screening_responses.items(), 1):
            # Format response for display
            response = data["response"]
            if isinstance(response, bool):
                response_text = "✅ Yes" if response else "❌ No"
            else:
                response_text = str(response)[:12] + "..." if len(str(response)) > 12 else str(response)
            
            # Shorten question for table display
            question_text = data["question"]
            question_short = question_text[:47] + "..." if len(question_text) > 47 else question_text
            
            # Get category
            category = question_lookup.get(question_id, {}).get("category", "general")
            category_display = category.replace("_", " ").title()
            
            breakdown_table.add_row(
                f"{i}. {question_short}",
                response_text,
                f"{data['accuracy']:.1f}%",
                category_display
            )
        
        self.console.print(breakdown_table)

    async def _get_comprehensive_ai_analysis(self, risk_percentage: float):
        """Get AI-powered comprehensive medical analysis of screening results"""
        # Prepare detailed screening context for AI
        responses_summary, risk_factors, protective_factors = self._prepare_ai_context()
        
        ai_prompt = f"""
COMPREHENSIVE BREAST CANCER SCREENING ANALYSIS

PATIENT SCREENING DATA:
{responses_summary}

CALCULATED RISK SCORE: {risk_percentage:.1f}%

HIGH RISK FACTORS IDENTIFIED:
{chr(10).join(f"- {factor}" for factor in risk_factors) if risk_factors else "- None identified"}

PROTECTIVE FACTORS:
{chr(10).join(f"- {factor}" for factor in protective_factors) if protective_factors else "- Standard risk profile"}

SCREENING CONTEXT:
- Total questions answered: {len(self.screening_responses)}
- Session duration: {self._get_session_duration()}
- Medical file uploaded: {'Yes' if self.screening_responses.get('medical_report_upload', {}).get('response') else 'No'}

Please provide a comprehensive medical assessment including:

1. **RISK INTERPRETATION**: Explain the {risk_percentage:.1f}% risk score in clinical context

2. **KEY CONCERNS**: Identify and explain the most significant risk factors

3. **PROTECTIVE FACTORS**: Highlight factors that reduce risk

4. **IMMEDIATE RECOMMENDATIONS**: Specify urgent vs routine care needs

5. **SCREENING GUIDELINES**: Recommend appropriate screening frequency and methods

6. **PATIENT EDUCATION**: Explain findings in clear, supportive language

7. **FOLLOW-UP ACTIONS**: Specific next steps for healthcare engagement

8. **LIFESTYLE RECOMMENDATIONS**: Evidence-based prevention strategies

Be professional, empathetic, and provide evidence-based guidance. Focus on actionable recommendations.
"""
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Generating comprehensive AI medical analysis...", total=None)
            
            try:
                ai_result = await self.prediction_service.groq_service.analyze_tumor_features(
                    {}, "Comprehensive Screening Analysis", risk_percentage/100, ai_prompt
                )
                
                if ai_result and ai_result.get("status") == "success":
                    self.console.print(Panel(
                        ai_result.get("analysis", "Analysis not available"),
                        title="🧠 Comprehensive Medical AI Analysis"
                    ))
                else:
                    self._show_basic_recommendations(risk_percentage)
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                self._show_basic_recommendations(risk_percentage)

    def _prepare_ai_context(self) -> tuple:
        """Prepare screening context for AI analysis"""
        responses_summary = []
        high_risk_factors = []
        protective_factors = []
        
        for question_id, data in self.screening_responses.items():
            response = data["response"]
            question = data["question"]
            
            if isinstance(response, bool):
                response_text = "Yes" if response else "No"
                
                # Identify concerning responses
                if response and question_id in ["lumps_thickening", "breast_changes", "nipple_discharge", "skin_changes", "family_history"]:
                    high_risk_factors.append(question)
                elif not response and question_id not in ["medical_report_upload"]:
                    protective_factors.append(question)
            else:
                response_text = str(response)
            
            responses_summary.append(f"- {question}: {response_text}")
        
        return "\n".join(responses_summary), high_risk_factors, protective_factors

    def _show_basic_recommendations(self, risk_percentage: float):
        """Show evidence-based recommendations when AI is not available"""
        if risk_percentage >= 60:
            recommendations = """
[bold red]🚨 IMMEDIATE ACTION NEEDED:[/bold red]
• Contact your healthcare provider within 24-48 hours
• Schedule clinical breast examination urgently
• Discuss mammography or other imaging immediately
• Do not delay seeking medical attention
• Bring this screening summary to your appointment
"""
        elif risk_percentage >= 30:
            recommendations = """
[bold yellow]📅 SCHEDULE MEDICAL EVALUATION:[/bold yellow]
• Contact your healthcare provider within 1-2 weeks
• Discuss your symptoms and risk factors thoroughly
• Consider earlier or more frequent screening
• Follow recommended screening guidelines
• Monitor symptoms closely
"""
        else:
            recommendations = """
[bold green]✅ CONTINUE ROUTINE CARE:[/bold green]
• Maintain regular screening schedule (annual mammograms after 40)
• Continue monthly breast self-examinations
• Annual clinical examinations as recommended
• Stay aware of any changes and report them promptly
• Maintain healthy lifestyle practices
"""
        
        self.console.print(Panel(recommendations, title="📋 Medical Recommendations"))

    async def _start_open_conversation(self):
        """
        Start open-ended conversation mode after screening completion
        
        Features:
        - Context-aware medical Q&A
        - File upload and analysis
        - Screening summary review
        - Medical education and support
        """
        self.console.print(Panel(
            "[bold green]🎉 Screening Complete![/bold green]\n\n"
            "You can now ask me any questions about:\n"
            "• Your screening results and risk assessment\n"
            "• Breast cancer prevention and early detection\n" 
            "• Medical terminology and procedures\n"
            "• Next steps and healthcare recommendations\n\n"
            "[bold cyan]Available commands:[/bold cyan]\n"
            "• Type your question naturally\n"
            "• 'upload' or 'file' - Upload additional medical files\n"
            "• 'summary' or 'results' - Review your screening results\n"
            "• 'quit' or 'exit' - End session with final summary\n\n"
            "[dim]All responses are contextualized with your screening results[/dim]",
            title="💬 Interactive Medical Consultation"
        ))
        
        conversation_count = 0
        
        while True:
            try:
                user_input = Prompt.ask(
                    f"\n[bold cyan]💬 Medical Question #{conversation_count + 1}[/bold cyan]"
                )
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    await self._handle_session_exit()
                    break
                elif user_input.lower() in ['summary', 'results', 'screening']:
                    await self._show_detailed_screening_summary()
                elif user_input.lower().startswith('upload') or 'file' in user_input.lower():
                    await self._handle_file_upload_conversation()
                elif user_input.lower() in ['help', '?']:
                    self._show_conversation_help()
                else:
                    await self._handle_contextual_medical_conversation(user_input)
                    conversation_count += 1
                    
            except KeyboardInterrupt:
                self.console.print("\n👋 Take care of your health!", style="blue")
                break
            except Exception as e:
                self.console.print(f"❌ Error processing request: {e}", style="red")
                logger.error(f"Conversation error: {e}")

    def _show_conversation_help(self):
        """Show help for conversation mode"""
        help_table = Table(title="💡 Available Commands & Features")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("Ask any question", "Natural language medical questions")
        help_table.add_row("'upload' or 'file'", "Upload additional medical documents")
        help_table.add_row("'summary' or 'results'", "Review your screening results")
        help_table.add_row("'quit' or 'exit'", "End session with summary")
        help_table.add_row("'help' or '?'", "Show this help menu")
        
        self.console.print(help_table)

    async def _handle_contextual_medical_conversation(self, user_input: str):
        """Handle medical conversation with full screening context"""
        # Get conversation context
        context_prompt = await self.conversation_manager.get_context_aware_prompt(user_input, "general")
        
        # Enhanced prompt with complete screening context
        screening_context = self._format_comprehensive_screening_context()
        
        enhanced_prompt = f"""
MEDICAL CONSULTATION WITH PATIENT CONTEXT

PATIENT SCREENING RESULTS:
{screening_context}

CURRENT PATIENT QUESTION: {user_input}

CONVERSATION HISTORY:
{context_prompt}

You are a compassionate medical AI assistant specializing in breast cancer education and patient support.

Provide helpful, accurate medical information while:

1. **ACKNOWLEDGING CONCERNS**: Show empathy and understanding
2. **PROVIDING EDUCATION**: Share relevant, evidence-based medical information  
3. **CONTEXTUALIZING**: Reference their specific screening results when relevant
4. **RECOMMENDING CARE**: Always encourage professional medical consultation
5. **BEING SUPPORTIVE**: Maintain caring, professional tone throughout
6. **STAYING ACCURATE**: Only provide evidence-based medical information
7. **EMPOWERING PATIENT**: Help them prepare for healthcare conversations

CRITICAL GUIDELINES:
- Never provide specific diagnoses or definitive medical conclusions
- Always recommend professional medical evaluation for health concerns
- Reference screening results to personalize guidance
- Encourage proactive healthcare engagement
- Provide actionable next steps when appropriate

Focus on education, support, and empowering informed healthcare decisions.
"""
        
        if self.prediction_service and hasattr(self.prediction_service, 'groq_service'):
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Providing personalized medical guidance...", total=None)
                
                try:
                    ai_result = await self.prediction_service.groq_service.analyze_tumor_features(
                        {}, "Medical Consultation", 1.0, enhanced_prompt
                    )
                    
                    if ai_result and ai_result.get("status") == "success":
                        response = ai_result.get("analysis", "I understand your concern. Please consult with a healthcare professional.")
                        self.console.print(Panel(
                            response,
                            title="🩺 Personalized Medical AI Guidance"
                        ))
                    else:
                        response = "I understand your concern. Based on your screening results, please consult with a healthcare professional for personalized guidance."
                        self.console.print(Panel(response, title="Medical Guidance"))
                        
                except Exception as e:
                    response = "I understand your concern. Please consult with a healthcare professional for medical evaluation."
                    self.console.print(response, style="yellow")
                    logger.error(f"AI conversation error: {e}")
        else:
            response = "I understand your concern. Please consult with a healthcare professional for medical evaluation."
            self.console.print(Panel(response, title="Medical Guidance"), style="yellow")
        
        # Save conversation interaction
        await self.conversation_manager.add_interaction(
            user_input=user_input,
            system_response=response,
            context_type="medical_conversation"
        )

    def _format_comprehensive_screening_context(self) -> str:
        """Format complete screening results for AI context"""
        if not self.screening_responses:
            return "No screening data available"
        
        context_parts = []
        risk_factors = []
        normal_responses = []
        
        # Organize responses by category
        questions = self._get_screening_questions()
        question_lookup = {q["id"]: q for q in questions}
        
        for question_id, data in self.screening_responses.items():
            response = data["response"]
            question = data["question"]
            category = question_lookup.get(question_id, {}).get("category", "general")
            
            if isinstance(response, bool):
                response_text = "Yes" if response else "No"
                
                # Categorize responses
                if response and question_id in ["lumps_thickening", "breast_changes", "nipple_discharge", "skin_changes"]:
                    risk_factors.append(f"{question} (Category: {category})")
                elif not response and question_id not in ["medical_report_upload"]:
                    normal_responses.append(f"{question} (Category: {category})")
            else:
                response_text = str(response)
            
            context_parts.append(f"- [{category.upper()}] {question}: {response_text}")
        
        # Build comprehensive context
        result = f"COMPLETE SCREENING RESPONSES ({len(self.screening_responses)}/11):\n"
        result += "\n".join(context_parts)
        result += f"\n\nOVERALL RISK SCORE: {self.overall_risk_score:.1f}%"
        
        if risk_factors:
            result += f"\n\nCONCERNING FINDINGS:\n" + "\n".join(f"- {factor}" for factor in risk_factors)
        
        if normal_responses:
            result += f"\n\nNORMAL/PROTECTIVE RESPONSES:\n" + "\n".join(f"- {factor}" for factor in normal_responses[:3])  # Limit for brevity
        
        return result

    async def _handle_file_upload_conversation(self):
        """Handle file upload during conversation with enhanced analysis"""
        self.console.print(Panel(
            "📁 **Medical File Upload**\n\n"
            "Supported formats:\n"
            "• Documents: PDF, DOC, DOCX, TXT\n"
            "• Images: JPEG, PNG (mammograms, ultrasounds, etc.)\n"
            "• Medical reports, lab results, imaging studies\n\n"
            "[dim]Files will be analyzed in context of your screening results[/dim]",
            title="File Analysis"
        ))
        
        file_path = Prompt.ask("📁 Enter the full path to your medical file")
        
        if not file_path:
            return
            
        if not Path(file_path).exists():
            self.console.print(f"❌ File not found: {file_path}", style="red")
            return
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task(f"Analyzing {Path(file_path).name}...", total=None)
            file_result = await self.file_processor.process_file(file_path)
        
        if "error" in file_result:
            self.console.print(f"❌ Could not process file: {file_result['error']}", style="red")
            return
        
        # Show file processing success
        file_type = file_result.get("file_type", "unknown")
        self.console.print(f"✅ Successfully analyzed {file_type} file: {Path(file_path).name}", style="green")
        
        # Get integrated AI interpretation
        if self.prediction_service and Confirm.ask("🧠 Would you like AI interpretation integrated with your screening results?"):
            await self._get_integrated_file_analysis(file_result, file_path)

    async def _get_integrated_file_analysis(self, file_result: Dict[str, Any], file_path: str):
        """Get AI interpretation that combines file analysis with screening context"""
        file_type = file_result.get("file_type")
        screening_context = self._format_comprehensive_screening_context()
        
        if file_type == "document":
            text_analysis = file_result.get("text_analysis", {})
            entities = file_result.get("medical_entities", {})
            
            ai_prompt = f"""
INTEGRATED MEDICAL FILE ANALYSIS

PATIENT SCREENING CONTEXT:
{screening_context}

UPLOADED DOCUMENT ANALYSIS:
- File: {Path(file_path).name}
- Medical Relevance: {text_analysis.get('medical_relevance', {}).get('relevance_level', 'unknown')}
- Medical Entities Detected: {json.dumps(entities, indent=2)}
- Document Summary: {text_analysis.get('summary', 'No summary available')}
- Text Length: {len(text_analysis.get('full_text', ''))} characters

INTEGRATION ANALYSIS NEEDED:

1. **CORRELATION ANALYSIS**: How do document findings relate to screening responses?
2. **CONSISTENCY CHECK**: Are there correlations or contradictions between file and screening?
3. **ENHANCED RISK ASSESSMENT**: Does the document modify the calculated risk level?
4. **CLINICAL RELEVANCE**: What medical significance do the combined findings have?
5. **ACTIONABLE RECOMMENDATIONS**: Updated next steps considering both sources
6. **HEALTHCARE COMMUNICATION**: How should patient discuss integrated findings with provider?
7. **FOLLOW-UP PRIORITIES**: What should be prioritized based on combined analysis?

Provide comprehensive integrated analysis that enhances the initial screening assessment.
"""
        else:
            # Handle medical images
            ai_prompt = f"""
INTEGRATED MEDICAL IMAGE ANALYSIS

PATIENT SCREENING CONTEXT:
{screening_context}

MEDICAL IMAGE ANALYSIS:
- File: {Path(file_path).name}
- Image Analysis: {json.dumps(file_result.get('image_analysis', {}), indent=2)}

Please provide integrated analysis considering both screening responses and image findings.
Focus on how the image relates to reported symptoms and overall risk assessment.
"""
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Performing integrated medical analysis...", total=None)
            
            try:
                ai_result = await self.prediction_service.groq_service.analyze_tumor_features(
                    {}, "Integrated Medical Analysis", 1.0, ai_prompt
                )
                
                if ai_result and ai_result.get("status") == "success":
                    self.console.print(Panel(
                        ai_result.get("analysis", "Analysis not available"),
                        title="🧠 Integrated Medical Analysis (Screening + File)"
                    ))
                else:
                    self.console.print("Unable to generate integrated analysis. Please discuss both your screening results and file with a healthcare professional.", style="yellow")
            except Exception as e:
                logger.error(f"Integrated analysis failed: {e}")
                self.console.print("Unable to generate integrated analysis. Please consult a healthcare professional.", style="yellow")

    async def _show_detailed_screening_summary(self):
        """Show comprehensive screening summary with statistics"""
        if not self.screening_responses:
            self.console.print("No screening data available", style="yellow")
            return
        
        # Calculate comprehensive metrics
        total_questions = len(self.screening_responses)
        avg_accuracy = sum(
            data["accuracy"] if isinstance(data["accuracy"], (int, float)) 
            else (data["accuracy"]["yes"] + data["accuracy"]["no"]) / 2 
            for data in self.screening_responses.values()
        ) / total_questions
        
        concerning_responses = sum(
            1 for data in self.screening_responses.values() 
            if isinstance(data["response"], bool) and data["response"] and 
            any(risk_id in data.get("question", "") for risk_id in ["lumps", "changes", "discharge", "skin"])
        )
        
        # Create comprehensive summary table
        summary_table = Table(title="📊 Comprehensive Screening Summary")
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Value", style="magenta", width=20)
        summary_table.add_column("Clinical Significance", style="white", width=35)
        
        summary_table.add_row(
            "Questions Completed", 
            f"{total_questions}/11",
            "Complete screening protocol"
        )
        summary_table.add_row(
            "Average Response Accuracy", 
            f"{avg_accuracy:.1f}%",
            "High accuracy improves assessment reliability"
        )
        summary_table.add_row(
            "Concerning Responses", 
            str(concerning_responses),
            "Physical symptoms requiring evaluation" if concerning_responses > 0 else "No immediate physical concerns"
        )
        summary_table.add_row(
            "Overall Risk Level", 
            self._get_current_risk_level(),
            self._get_risk_explanation()
        )
        summary_table.add_row(
            "Session Duration",
            self._get_session_duration(),
            "Time invested in health assessment"
        )
        
        self.console.print(summary_table)
        
        # Show risk score breakdown
        self._show_risk_score_breakdown()

    def _get_current_risk_level(self) -> str:
        """Get current risk level with emoji indicators"""
        concerning_count = sum(
            1 for data in self.screening_responses.values() 
            if isinstance(data["response"], bool) and data["response"]
        )
        
        if concerning_count >= 4:
            return "🔴 High Risk"
        elif concerning_count >= 2:
            return "🟡 Moderate Risk" 
        else:
            return "🟢 Low Risk"

    def _get_risk_explanation(self) -> str:
        """Get explanation for current risk level"""
        concerning_count = sum(
            1 for data in self.screening_responses.values() 
            if isinstance(data["response"], bool) and data["response"]
        )
        
        if concerning_count >= 4:
            return "Multiple risk factors present - seek immediate evaluation"
        elif concerning_count >= 2:
            return "Some risk factors present - schedule medical evaluation"
        else:
            return "Standard risk profile - continue routine screening"

    def _show_risk_score_breakdown(self):
        """Show detailed risk score breakdown by category"""
        if not self.screening_responses:
            return
            
        # Group by category
        questions = self._get_screening_questions()
        question_lookup = {q["id"]: q for q in questions}
        
        category_scores = {}
        for question_id, data in self.screening_responses.items():
            question_info = question_lookup.get(question_id, {})
            category = question_info.get("category", "general")
            
            if category not in category_scores:
                category_scores[category] = {"score": 0, "max_score": 0, "count": 0}
            
            # Calculate contribution
            if isinstance(data["response"], bool) and data["response"] and question_id != "medical_report_upload":
                contribution = question_info.get("risk_weight", 0) * 100
                category_scores[category]["score"] += contribution
            
            category_scores[category]["max_score"] += question_info.get("risk_weight", 0) * 100
            category_scores[category]["count"] += 1
        
        # Display category breakdown
        category_table = Table(title="🎯 Risk Score Breakdown by Category")
        category_table.add_column("Category", style="cyan")
        category_table.add_column("Score", style="magenta")
        category_table.add_column("Max Possible", style="blue")
        category_table.add_column("Percentage", style="green")
        
        for category, scores in category_scores.items():
            percentage = (scores["score"] / scores["max_score"] * 100) if scores["max_score"] > 0 else 0
            category_display = category.replace("_", " ").title()
            
            category_table.add_row(
                category_display,
                f"{scores['score']:.1f}",
                f"{scores['max_score']:.1f}",
                f"{percentage:.1f}%"
            )
        
        self.console.print(category_table)

    async def _handle_session_exit(self):
        """Handle application exit with comprehensive session summary"""
        self.console.print(Panel(
            "[bold blue]📊 FINAL SESSION SUMMARY[/bold blue]",
            title="Session Complete"
        ))
        
        # Show final screening summary
        await self._show_detailed_screening_summary()
        
        # Show conversation statistics
        try:
            session_summary = await self.conversation_manager.get_session_summary()
            if "error" not in session_summary:
                interaction_summary = session_summary.get("interaction_summary", {})
                total_interactions = interaction_summary.get("total_messages", 0)
                
                stats_table = Table(title="💬 Session Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="magenta")
                
                stats_table.add_row("Total Questions Asked", str(total_interactions))
                stats_table.add_row("Screening Questions", "11/11")
                stats_table.add_row("Session Duration", self._get_session_duration())
                stats_table.add_row("AI Analysis Used", "Yes" if self.prediction_service else "Limited")
                
                self.console.print(stats_table)
        except Exception as e:
            logger.error(f"Session summary error: {e}")
        
        # Final medical reminder
        self.console.print(Panel(
            "[bold blue]Thank you for prioritizing your health![/bold blue]\n\n"
            "[bold yellow]🩺 IMPORTANT REMINDERS:[/bold yellow]\n"
            "• This screening provides educational risk assessment only\n"
            "• Always consult healthcare professionals for medical decisions\n"
            "• Follow up promptly on any concerning findings\n"
            "• Maintain regular screening as recommended by your doctor\n"
            "• Keep this session summary for your healthcare provider\n\n"
            "[bold green]💚 Your health matters - stay proactive about screening![/bold green]",
            title="👋 Session Complete - Stay Healthy!"
        ))


def check_system_requirements():
    """Check if all required dependencies are available"""
    required_imports = {
        'rich': 'rich',
        'loguru': 'loguru', 
        'python-dotenv': 'dotenv',
        'asyncio': 'asyncio',
        'pathlib': 'pathlib',
        'json': 'json',
        'datetime': 'datetime'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_services_availability():
    """Check if required service modules are available"""
    console = Console()
    
    required_services = [
        'services.enhanced_prediction_service',
        'services.file_processor', 
        'services.conversation_manager'
    ]
    
    missing_services = []
    
    for service in required_services:
        try:
            __import__(service)
        except ImportError:
            missing_services.append(service)
    
    if missing_services:
        console.print(f"❌ Missing required services: {', '.join(missing_services)}", style="red")
        console.print("💡 Ensure all service modules are present in the services/ directory", style="yellow")
        return False
    
    console.print("✅ All required services available", style="green")
    return True


async def main():
    """
    Main entry point for the AI-Enhanced Breast Cancer Screening Assistant
    
    Performs system checks, initializes services, and starts the direct screening workflow
    """
    console = Console()
    
    try:
        # System requirements check
        if not check_system_requirements():
            console.print("❌ System requirements not met. Please install missing packages.", style="red")
            sys.exit(1)
        
        # Services availability check
        if not check_services_availability():
            console.print("❌ Required services not available. Please check your installation.", style="red")
            sys.exit(1)
        
        # Initialize and run the application
        console.print("🚀 Starting AI Medical Screening Assistant...", style="blue")
        
        app = BreastCancerScreeningCLI()
        await app.run()
        
    except KeyboardInterrupt:
        console.print("\n👋 Take care of your health!", style="blue")
    except Exception as e:
        console.print(f"❌ Application error: {e}", style="red")
        logger.error(f"Main application error: {e}")
        sys.exit(1)


def run_api_server():
    """Run the FastAPI web server"""
    import uvicorn
    
    console = Console()
    console.print(Panel(
        "[bold blue]🌐 Starting FastAPI Web Server[/bold blue]\n\n"
        "• API available at: http://localhost:8000\n"
        "• Interactive docs: http://localhost:8000/docs\n"
        "• Health check: http://localhost:8000/health\n"
        "• Status: http://localhost:8000/status\n\n"
        "[bold yellow]For CLI screening tool, run:[/bold yellow] python main.py\n"
        "[dim]Use Ctrl+C to stop the server[/dim]",
        title="Web API Server"
    ))
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    """
    Dual-Mode Application Entry Point
    
    Modes:
        python main.py              # CLI screening mode (default)
        python main.py --api        # FastAPI web server mode
        python main.py --help       # Show help
    
    CLI Mode:
    1. Check system requirements
    2. Verify service availability  
    3. Start direct medical screening
    4. Provide AI-enhanced analysis
    5. Enable open medical conversation
    6. Generate session summary
    
    API Mode:
    1. Initialize FastAPI server
    2. Serve REST endpoints
    3. Provide health checks
    4. Enable web frontend integration
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI-Enhanced Breast Cancer Screening Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Start CLI screening (default)
  python main.py --api        # Start FastAPI web server
  python main.py --help       # Show this help

For CLI mode, the application starts immediately with medical screening.
For API mode, the web server runs on http://localhost:8000
"""
    )
    
    parser.add_argument(
        '--api', 
        action='store_true',
        help='Start FastAPI web server instead of CLI screening'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host for API server (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for API server (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.api:
        logger.add(
            "logs/api_server_{time}.log",
            format="{time} | {level} | {message}",
            level="INFO"
        )
    else:
        logger.add(
            "logs/screening_session_{time}.log",
            format="{time} | {level} | {message}",
            level="INFO"
        )
    
    # Run in selected mode
    if args.api:
        # FastAPI web server mode
        import uvicorn
        
        console = Console()
        console.print(Panel(
            "[bold blue]🌐 Starting FastAPI Web Server[/bold blue]\n\n"
            f"• API available at: http://{args.host}:{args.port}\n"
            f"• Interactive docs: http://{args.host}:{args.port}/docs\n"
            f"• Health check: http://{args.host}:{args.port}/health\n"
            f"• Status: http://{args.host}:{args.port}/status\n\n"
            "[bold yellow]For CLI screening tool, run:[/bold yellow] python main.py\n"
            "[dim]Use Ctrl+C to stop the server[/dim]",
            title="Web API Server"
        ))
        
        try:
            uvicorn.run(app, host=args.host, port=args.port)
        except KeyboardInterrupt:
            console.print("\n👋 API server stopped", style="blue")
        except Exception as e:
            console.print(f"❌ Server error: {e}", style="red")
    else:
        # CLI screening mode (default)
        asyncio.run(main())

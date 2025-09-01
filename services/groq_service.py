"""
Groq API Service for breast cancer prediction analysis
Uses Llama3-8B model with Apache2 license
"""

import os
from typing import Optional, Dict, Any
from groq import Groq
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Groq service initialized with model: {self.model_name}")

    async def analyze_tumor_features(self, features: Dict[str, float], ml_prediction: str, confidence: float, rag_context: str) -> Optional[Dict[str, Any]]:
        """
        Analyze tumor features using Groq AI and provide enhanced medical interpretation
        
        Args:
            features: Dictionary of tumor measurements
            ml_prediction: Machine learning model prediction (Benign/Malignant)
            confidence: Confidence score from ML model
            rag_context: Relevant medical knowledge from RAG system
        
        Returns:
            Enhanced analysis with explanation and recommendations
        """
        try:
            # Format features for analysis
            features_text = self._format_features(features)
            
            prompt = f"""
You are a medical AI assistant specialized in breast cancer diagnosis analysis. 

TUMOR MEASUREMENTS:
{features_text}

MACHINE LEARNING PREDICTION: {ml_prediction} (Confidence: {confidence:.2%})

RELEVANT MEDICAL KNOWLEDGE:
{rag_context}

Please provide a comprehensive medical analysis including:

1. CLINICAL INTERPRETATION: Interpret the tumor measurements in medical context
2. RISK ASSESSMENT: Evaluate the risk factors based on the measurements
3. FEATURE ANALYSIS: Highlight the most concerning or reassuring features
4. MEDICAL REASONING: Explain why these measurements suggest benign or malignant characteristics
5. RECOMMENDATIONS: Provide appropriate medical recommendations
6. CONFIDENCE ASSESSMENT: Evaluate the reliability of the prediction based on feature patterns

Keep the response professional, medically accurate, and suitable for healthcare providers.
Format the response in clear sections for easy interpretation.
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a specialized medical AI assistant for breast cancer diagnosis analysis. Provide accurate, professional medical interpretations based on tumor measurements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent medical analysis
                max_tokens=1500,
                top_p=0.9
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "provider": "groq",
                "model": self.model_name,
                "analysis": analysis,
                "enhanced_prediction": ml_prediction,
                "confidence_score": confidence,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return None

    def _format_features(self, features: Dict[str, float]) -> str:
        """Format tumor features for AI analysis"""
        formatted = []
        
        # Group features by category
        worst_features = {k: v for k, v in features.items() if "worst" in k}
        mean_features = {k: v for k, v in features.items() if "mean" in k}
        
        formatted.append("WORST MEASUREMENTS:")
        for feature, value in worst_features.items():
            formatted.append(f"  {feature.replace('_', ' ').title()}: {value:.4f}")
        
        formatted.append("\nMEAN MEASUREMENTS:")
        for feature, value in mean_features.items():
            formatted.append(f"  {feature.replace('_', ' ').title()}: {value:.4f}")
        
        return "\n".join(formatted)

    async def get_model_info(self) -> Dict[str, str]:
        """Get information about the current Groq model"""
        return {
            "provider": "Groq",
            "model": self.model_name,
            "license": "Apache 2.0",
            "description": "Llama3 8B parameter model optimized for fast inference"
        }

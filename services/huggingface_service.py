"""
HuggingFace API Service for breast cancer prediction analysis
Uses Apache2 licensed models as secondary fallback
"""

import os
import requests
from typing import Optional, Dict, Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceService:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        self.model_name = os.getenv("HF_MODEL_NAME", "microsoft/BioGPT-Large")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"HuggingFace service initialized with model: {self.model_name}")

    async def analyze_tumor_features(self, features: Dict[str, float], ml_prediction: str, confidence: float, rag_context: str) -> Optional[Dict[str, Any]]:
        """
        Analyze tumor features using HuggingFace AI as fallback service
        
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
            
            prompt = f"""Medical Analysis Request:

Tumor Measurements: {features_text}
ML Prediction: {ml_prediction} (Confidence: {confidence:.2%})
Medical Context: {rag_context[:500]}...

Provide medical interpretation focusing on:
1. Clinical significance of measurements
2. Risk assessment based on tumor characteristics
3. Medical recommendations for patient care

Analysis:"""

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 800,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    analysis = result[0].get("generated_text", "").replace(prompt, "").strip()
                elif isinstance(result, dict):
                    analysis = result.get("generated_text", "").replace(prompt, "").strip()
                else:
                    analysis = "Unable to generate analysis from HuggingFace model"
                
                return {
                    "provider": "huggingface",
                    "model": self.model_name,
                    "analysis": analysis,
                    "enhanced_prediction": ml_prediction,
                    "confidence_score": confidence,
                    "status": "success"
                }
            else:
                logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"HuggingFace service error: {str(e)}")
            return None

    def _format_features(self, features: Dict[str, float]) -> str:
        """Format tumor features for AI analysis"""
        # Create concise feature summary for HuggingFace
        key_features = []
        
        # Highlight most important features
        important_features = [
            "radius_worst", "perimeter_worst", "area_worst", 
            "concave_points_worst", "concavity_worst", "compactness_worst"
        ]
        
        for feature in important_features:
            if feature in features:
                key_features.append(f"{feature.replace('_', ' ')}: {features[feature]:.3f}")
        
        return ", ".join(key_features)

    async def get_model_info(self) -> Dict[str, str]:
        """Get information about the current HuggingFace model"""
        return {
            "provider": "HuggingFace",
            "model": self.model_name,
            "license": "Apache 2.0",
            "description": "BioGPT model specialized for biomedical text generation"
        }

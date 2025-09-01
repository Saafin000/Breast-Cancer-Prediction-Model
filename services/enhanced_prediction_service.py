"""
Enhanced Prediction Service that combines:
1. Traditional ML model (Random Forest)
2. AI analysis (Groq primary, HuggingFace fallback)  
3. RAG-based medical knowledge retrieval
"""

import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

from .groq_service import GroqService
from .huggingface_service import HuggingFaceService
from .rag_service import RAGService

load_dotenv()

class EnhancedPredictionService:
    def __init__(self, model_path: str = "cancer_model.pkl"):
        # Load traditional ML model
        self.ml_model = joblib.load(model_path)
        
        # Initialize AI services
        try:
            self.groq_service = GroqService()
            logger.info("Groq service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq service: {e}")
            self.groq_service = None
        
        try:
            self.hf_service = HuggingFaceService()
            logger.info("HuggingFace service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace service: {e}")
            self.hf_service = None
        
        # Initialize RAG service
        try:
            self.rag_service = RAGService()
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG service: {e}")
            self.rag_service = None
        
        # Feature names in correct order
        self.feature_names = [
            "radius_worst", "perimeter_worst", "area_worst", "concave_points_worst",
            "concavity_worst", "compactness_worst", "radius_mean", "perimeter_mean",
            "area_mean", "concave_points_mean", "concavity_mean", "compactness_mean",
            "texture_worst", "smoothness_worst", "symmetry_worst"
        ]

    async def predict_enhanced(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform enhanced prediction combining ML model, AI analysis, and RAG
        
        Args:
            features: Dictionary of tumor measurements
        
        Returns:
            Comprehensive prediction results with AI analysis
        """
        try:
            # Step 1: Traditional ML prediction
            ml_result = self._get_ml_prediction(features)
            
            # Step 2: Retrieve relevant medical knowledge
            rag_context = ""
            if self.rag_service:
                rag_context = await self.rag_service.retrieve_relevant_knowledge(features)
            
            # Step 3: AI-enhanced analysis
            ai_analysis = await self._get_ai_analysis(features, ml_result, rag_context)
            
            # Step 4: Risk assessment
            risk_assessment = self._calculate_risk_assessment(features)
            
            # Combine all results
            enhanced_result = {
                "ml_prediction": ml_result["prediction"],
                "ml_confidence": ml_result["confidence"],
                "ai_analysis": ai_analysis,
                "risk_assessment": risk_assessment,
                "feature_analysis": self._analyze_key_features(features),
                "medical_context": rag_context[:500] + "..." if len(rag_context) > 500 else rag_context,
                "timestamp": datetime.now().isoformat(),
                "services_used": self._get_services_status()
            }
            
            logger.info(f"Enhanced prediction completed: {ml_result['prediction']}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {str(e)}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "ml_prediction": "Error",
                "ml_confidence": 0.0
            }

    def _get_ml_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Get prediction from traditional ML model"""
        try:
            # Convert features to array in correct order
            input_data = np.array([[features[name] for name in self.feature_names]])
            
            # Get prediction and probability
            prediction = self.ml_model.predict(input_data)[0]
            probabilities = self.ml_model.predict_proba(input_data)[0]
            
            result = "Malignant" if prediction == 1 else "Benign"
            confidence = max(probabilities)
            
            return {
                "prediction": result,
                "confidence": confidence,
                "probabilities": {
                    "benign": probabilities[0],
                    "malignant": probabilities[1]
                }
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "probabilities": {"benign": 0.0, "malignant": 0.0}
            }

    async def _get_ai_analysis(self, features: Dict[str, float], ml_result: Dict[str, Any], rag_context: str) -> Optional[Dict[str, Any]]:
        """Get AI analysis from Groq (primary) or HuggingFace (fallback)"""
        # Try Groq first
        if self.groq_service:
            result = await self.groq_service.analyze_tumor_features(
                features, ml_result["prediction"], ml_result["confidence"], rag_context
            )
            if result:
                return result
        
        # Fallback to HuggingFace
        if self.hf_service:
            result = await self.hf_service.analyze_tumor_features(
                features, ml_result["prediction"], ml_result["confidence"], rag_context
            )
            if result:
                return result
        
        # If both fail, return basic analysis
        return {
            "provider": "fallback",
            "model": "basic_analysis",
            "analysis": f"Basic ML prediction: {ml_result['prediction']} with {ml_result['confidence']:.2%} confidence",
            "enhanced_prediction": ml_result["prediction"],
            "confidence_score": ml_result["confidence"],
            "status": "ai_services_unavailable"
        }

    def _calculate_risk_assessment(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Calculate detailed risk assessment based on feature thresholds"""
        risk_factors = []
        protective_factors = []
        
        # High-risk thresholds
        high_risk_checks = {
            "radius_worst": (25, "Large tumor radius"),
            "area_worst": (2000, "Large tumor area"),
            "concavity_worst": (0.5, "High concavity"),
            "concave_points_worst": (0.25, "Many concave points"),
            "texture_worst": (30, "High texture variation")
        }
        
        # Moderate-risk thresholds
        moderate_risk_checks = {
            "radius_worst": (15, "Moderate tumor radius"),
            "area_worst": (800, "Moderate tumor area"),
            "concavity_worst": (0.2, "Moderate concavity"),
            "compactness_worst": (0.3, "High compactness")
        }
        
        # Check high-risk factors
        for feature, (threshold, description) in high_risk_checks.items():
            if features.get(feature, 0) > threshold:
                risk_factors.append(f"{description} ({features[feature]:.3f} > {threshold})")
        
        # Check moderate-risk factors
        for feature, (threshold, description) in moderate_risk_checks.items():
            value = features.get(feature, 0)
            if threshold < value <= high_risk_checks.get(feature, (float('inf'), ""))[0]:
                risk_factors.append(f"{description} ({value:.3f})")
        
        # Check protective factors (low values)
        protective_checks = {
            "radius_worst": (15, "Small tumor radius"),
            "area_worst": (800, "Small tumor area"),
            "concavity_worst": (0.2, "Low concavity"),
            "symmetry_worst": (0.25, "Good symmetry")
        }
        
        for feature, (threshold, description) in protective_checks.items():
            if features.get(feature, 0) < threshold:
                protective_factors.append(f"{description} ({features[feature]:.3f} < {threshold})")
        
        # Calculate overall risk score
        risk_score = len(risk_factors) / (len(risk_factors) + len(protective_factors) + 1)
        
        if risk_score > 0.7:
            risk_level = "High"
        elif risk_score > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "total_factors_analyzed": len(high_risk_checks) + len(moderate_risk_checks)
        }

    def _analyze_key_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the most important features for interpretation"""
        # Feature importance ranking (based on Random Forest)
        feature_importance = {
            "concave_points_worst": 1,
            "perimeter_worst": 2,
            "concavity_worst": 3,
            "radius_worst": 4,
            "area_worst": 5
        }
        
        key_insights = {}
        for feature, rank in feature_importance.items():
            if feature in features:
                value = features[feature]
                key_insights[feature] = {
                    "value": value,
                    "importance_rank": rank,
                    "interpretation": self._interpret_feature_value(feature, value)
                }
        
        return key_insights

    def _interpret_feature_value(self, feature_name: str, value: float) -> str:
        """Interpret individual feature values"""
        interpretations = {
            "concave_points_worst": {
                "low": (0.15, "Few concave points - suggests benign characteristics"),
                "high": (0.25, "Many concave points - strong malignancy indicator")
            },
            "radius_worst": {
                "low": (15, "Small tumor radius - favorable"),
                "high": (25, "Large tumor radius - concerning")
            },
            "area_worst": {
                "low": (800, "Small tumor area - favorable"),
                "high": (2000, "Large tumor area - concerning")
            },
            "concavity_worst": {
                "low": (0.2, "Low concavity - suggests regular borders"),
                "high": (0.5, "High concavity - irregular malignant pattern")
            },
            "perimeter_worst": {
                "low": (100, "Small tumor perimeter - favorable"),
                "high": (150, "Large tumor perimeter - concerning")
            }
        }
        
        if feature_name in interpretations:
            thresholds = interpretations[feature_name]
            if value < thresholds["low"][0]:
                return thresholds["low"][1]
            elif value > thresholds["high"][0]:
                return thresholds["high"][1]
            else:
                return "Moderate values - requires clinical correlation"
        
        return "Standard measurement"

    def _get_services_status(self) -> Dict[str, bool]:
        """Get status of all services"""
        return {
            "ml_model": True,
            "groq_service": self.groq_service is not None,
            "huggingface_service": self.hf_service is not None,
            "rag_service": self.rag_service is not None
        }

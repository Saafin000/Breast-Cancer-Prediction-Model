"""
Conversation Manager for maintaining context-aware interactions
Handles conversation history, context retention, and smart responses
"""

import json
import jsonlines
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from loguru import logger

@dataclass_json
@dataclass
class ConversationMessage:
    timestamp: str
    user_input: str
    system_response: str
    context_type: str  # 'prediction', 'file_analysis', 'general'
    file_processed: Optional[str] = None
    prediction_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None

@dataclass_json
@dataclass
class ConversationContext:
    session_id: str
    start_time: str
    last_interaction: str
    total_interactions: int
    context_summary: str
    processed_files: List[str]
    recent_predictions: List[Dict[str, Any]]

class ConversationManager:
    def __init__(self, session_file: str = "conversation_history.jsonl"):
        self.session_file = Path(session_file)
        self.current_context = ConversationContext(
            session_id=self._generate_session_id(),
            start_time=datetime.now().isoformat(),
            last_interaction=datetime.now().isoformat(),
            total_interactions=0,
            context_summary="New conversation started",
            processed_files=[],
            recent_predictions=[]
        )
        self.conversation_history: List[ConversationMessage] = []
        self._load_previous_session()
        logger.info(f"Conversation manager initialized. Session ID: {self.current_context.session_id}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _load_previous_session(self):
        """Load previous conversation history if exists"""
        try:
            if self.session_file.exists():
                with jsonlines.open(self.session_file, 'r') as reader:
                    for line in reader:
                        if line.get("type") == "message":
                            msg = ConversationMessage.from_dict(line["data"])
                            self.conversation_history.append(msg)
                        elif line.get("type") == "context":
                            # Load most recent context
                            self.current_context = ConversationContext.from_dict(line["data"])
                
                logger.info(f"Loaded {len(self.conversation_history)} previous messages")
            else:
                logger.info("Starting new conversation session")
                
        except Exception as e:
            logger.warning(f"Could not load previous session: {e}")

    async def add_interaction(self, user_input: str, system_response: str, 
                            context_type: str = "general", **kwargs) -> None:
        """
        Add new interaction to conversation history
        
        Args:
            user_input: What the user said/asked
            system_response: System's response
            context_type: Type of interaction (prediction, file_analysis, general)
            **kwargs: Additional context data
        """
        try:
            message = ConversationMessage(
                timestamp=datetime.now().isoformat(),
                user_input=user_input,
                system_response=system_response,
                context_type=context_type,
                file_processed=kwargs.get("file_processed"),
                prediction_result=kwargs.get("prediction_result"),
                confidence_score=kwargs.get("confidence_score")
            )
            
            self.conversation_history.append(message)
            
            # Update context
            self.current_context.last_interaction = message.timestamp
            self.current_context.total_interactions += 1
            
            # Update context based on interaction type
            if context_type == "file_analysis" and kwargs.get("file_processed"):
                if kwargs["file_processed"] not in self.current_context.processed_files:
                    self.current_context.processed_files.append(kwargs["file_processed"])
            
            if context_type == "prediction" and kwargs.get("prediction_result"):
                self.current_context.recent_predictions.append({
                    "timestamp": message.timestamp,
                    "prediction": kwargs["prediction_result"],
                    "confidence": kwargs.get("confidence_score", 0)
                })
                # Keep only last 5 predictions
                self.current_context.recent_predictions = self.current_context.recent_predictions[-5:]
            
            # Update context summary
            self._update_context_summary()
            
            # Save to file
            await self._save_interaction(message)
            
        except Exception as e:
            logger.error(f"Error adding interaction: {str(e)}")

    def _update_context_summary(self):
        """Update conversation context summary"""
        recent_messages = self.conversation_history[-5:]  # Last 5 interactions
        
        summary_parts = []
        
        # Add file processing context
        if self.current_context.processed_files:
            file_types = set()
            for file_path in self.current_context.processed_files:
                ext = Path(file_path).suffix.lower()
                file_types.add(ext)
            summary_parts.append(f"Processed {len(self.current_context.processed_files)} files ({', '.join(file_types)})")
        
        # Add prediction context
        if self.current_context.recent_predictions:
            malignant_count = sum(1 for p in self.current_context.recent_predictions 
                                if p["prediction"].get("ml_prediction") == "Malignant")
            benign_count = len(self.current_context.recent_predictions) - malignant_count
            summary_parts.append(f"Made {len(self.current_context.recent_predictions)} predictions ({malignant_count} malignant, {benign_count} benign)")
        
        # Add recent interaction context
        if recent_messages:
            context_types = [msg.context_type for msg in recent_messages]
            unique_types = set(context_types)
            summary_parts.append(f"Recent interactions: {', '.join(unique_types)}")
        
        self.current_context.context_summary = "; ".join(summary_parts) if summary_parts else "Active conversation session"

    async def _save_interaction(self, message: ConversationMessage):
        """Save interaction to persistent storage"""
        try:
            with jsonlines.open(self.session_file, 'a') as writer:
                # Save message
                writer.write({
                    "type": "message",
                    "data": message.to_dict()
                })
                
                # Save updated context
                writer.write({
                    "type": "context", 
                    "data": self.current_context.to_dict()
                })
                
        except Exception as e:
            logger.error(f"Error saving interaction: {str(e)}")

    async def get_relevant_context(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """
        Get relevant conversation context for the current query
        
        Args:
            query: Current user query
            context_type: Type of context needed
            
        Returns:
            Relevant context information
        """
        try:
            context = {
                "session_info": {
                    "session_id": self.current_context.session_id,
                    "total_interactions": self.current_context.total_interactions,
                    "session_duration": self._calculate_session_duration(),
                    "context_summary": self.current_context.context_summary
                },
                "relevant_history": [],
                "file_context": [],
                "prediction_context": []
            }
            
            # Get relevant conversation history
            query_lower = query.lower()
            for message in self.conversation_history[-10:]:  # Last 10 messages
                if (any(word in message.user_input.lower() for word in query_lower.split()) or
                    any(word in message.system_response.lower() for word in query_lower.split())):
                    context["relevant_history"].append({
                        "timestamp": message.timestamp,
                        "user_input": message.user_input[:100] + "..." if len(message.user_input) > 100 else message.user_input,
                        "context_type": message.context_type
                    })
            
            # Add file context if relevant
            if "file" in query_lower or context_type == "file_analysis":
                context["file_context"] = [
                    {"file": f, "processed_at": "recent"} 
                    for f in self.current_context.processed_files[-3:]  # Last 3 files
                ]
            
            # Add prediction context if relevant
            if any(word in query_lower for word in ["predict", "cancer", "tumor", "malignant", "benign"]):
                context["prediction_context"] = self.current_context.recent_predictions[-3:]  # Last 3 predictions
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return {"error": f"Context retrieval failed: {str(e)}"}

    def _calculate_session_duration(self) -> str:
        """Calculate session duration in human-readable format"""
        try:
            start = datetime.fromisoformat(self.current_context.start_time)
            now = datetime.now()
            duration = now - start
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
                
        except:
            return "unknown"

    async def get_context_aware_prompt(self, current_query: str, context_type: str = "general") -> str:
        """
        Generate context-aware prompt for AI services
        
        Args:
            current_query: Current user query
            context_type: Type of interaction
            
        Returns:
            Enhanced prompt with conversation context
        """
        try:
            context = await self.get_relevant_context(current_query, context_type)
            
            prompt_parts = [f"Current Query: {current_query}"]
            
            # Add session context
            session_info = context.get("session_info", {})
            if session_info.get("total_interactions", 0) > 1:
                prompt_parts.append(f"Conversation Context: {session_info.get('context_summary', '')}")
            
            # Add relevant history
            relevant_history = context.get("relevant_history", [])
            if relevant_history:
                prompt_parts.append("Recent Relevant Interactions:")
                for hist in relevant_history[-2:]:  # Last 2 relevant
                    prompt_parts.append(f"- User asked: {hist['user_input']}")
            
            # Add file context
            file_context = context.get("file_context", [])
            if file_context:
                prompt_parts.append(f"Recently Processed Files: {', '.join([Path(f['file']).name for f in file_context])}")
            
            # Add prediction context
            prediction_context = context.get("prediction_context", [])
            if prediction_context:
                recent_pred = prediction_context[-1]
                pred_result = recent_pred.get("prediction", {}).get("ml_prediction", "Unknown")
                prompt_parts.append(f"Recent Prediction: {pred_result}")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error generating context-aware prompt: {str(e)}")
            return current_query

    async def clear_session(self):
        """Clear current session and start fresh"""
        try:
            # Save final context
            await self._save_interaction(ConversationMessage(
                timestamp=datetime.now().isoformat(),
                user_input="SESSION_END",
                system_response="Session cleared by user",
                context_type="system"
            ))
            
            # Reset context
            self.current_context = ConversationContext(
                session_id=self._generate_session_id(),
                start_time=datetime.now().isoformat(),
                last_interaction=datetime.now().isoformat(),
                total_interactions=0,
                context_summary="New conversation started",
                processed_files=[],
                recent_predictions=[]
            )
            self.conversation_history = []
            
            logger.info("Conversation session cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}")
            return False

    async def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        try:
            # Analyze conversation patterns
            context_types = [msg.context_type for msg in self.conversation_history]
            context_counts = {}
            for ctx_type in context_types:
                context_counts[ctx_type] = context_counts.get(ctx_type, 0) + 1
            
            # File analysis summary
            file_types_processed = {}
            for file_path in self.current_context.processed_files:
                ext = Path(file_path).suffix.lower()
                file_types_processed[ext] = file_types_processed.get(ext, 0) + 1
            
            # Prediction summary
            prediction_summary = {"total": len(self.current_context.recent_predictions)}
            if self.current_context.recent_predictions:
                malignant_count = sum(1 for p in self.current_context.recent_predictions 
                                    if p["prediction"].get("ml_prediction") == "Malignant")
                prediction_summary.update({
                    "malignant": malignant_count,
                    "benign": len(self.current_context.recent_predictions) - malignant_count,
                    "avg_confidence": sum(p.get("confidence", 0) for p in self.current_context.recent_predictions) / len(self.current_context.recent_predictions)
                })
            
            return {
                "session_info": asdict(self.current_context),
                "interaction_summary": {
                    "total_messages": len(self.conversation_history),
                    "context_type_breakdown": context_counts,
                    "session_duration": self._calculate_session_duration()
                },
                "file_processing_summary": {
                    "total_files": len(self.current_context.processed_files),
                    "file_types": file_types_processed
                },
                "prediction_summary": prediction_summary
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return {"error": f"Could not generate summary: {str(e)}"}

    async def search_conversation_history(self, search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history for specific terms"""
        try:
            search_term_lower = search_term.lower()
            matching_messages = []
            
            for message in self.conversation_history:
                if (search_term_lower in message.user_input.lower() or 
                    search_term_lower in message.system_response.lower()):
                    matching_messages.append({
                        "timestamp": message.timestamp,
                        "user_input": message.user_input,
                        "system_response": message.system_response[:200] + "..." if len(message.system_response) > 200 else message.system_response,
                        "context_type": message.context_type,
                        "relevance": "high" if search_term_lower in message.user_input.lower() else "medium"
                    })
            
            # Sort by relevance and recency
            matching_messages.sort(key=lambda x: (x["relevance"] == "high", x["timestamp"]), reverse=True)
            
            return matching_messages[:limit]
            
        except Exception as e:
            logger.error(f"Error searching conversation history: {str(e)}")
            return []

"""
RAG (Retrieval-Augmented Generation) Service for medical knowledge retrieval
Uses ChromaDB for vector storage and sentence-transformers for embeddings
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self):
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./rag_data/vector_store")
        self.knowledge_path = os.getenv("MEDICAL_KNOWLEDGE_PATH", "./rag_data/medical_knowledge.txt")
        
        # Initialize embedding model (Apache2 licensed)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        self.collection_name = "medical_knowledge"
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info("Loaded existing medical knowledge collection")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Breast cancer medical knowledge base"}
            )
            self._initialize_knowledge_base()
            logger.info("Created new medical knowledge collection")

    def _initialize_knowledge_base(self):
        """Initialize the vector database with medical knowledge"""
        try:
            # Read medical knowledge file
            if not os.path.exists(self.knowledge_path):
                logger.warning(f"Medical knowledge file not found: {self.knowledge_path}")
                return
            
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks by sections
            sections = content.split('== ')
            documents = []
            metadatas = []
            ids = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    # Extract section title and content
                    if ' ==' in section:
                        title, content = section.split(' ==', 1)
                        title = title.strip()
                    else:
                        title = f"Section {i}"
                        content = section
                    
                    # Further split large sections into smaller chunks
                    chunks = self._split_into_chunks(content.strip(), max_length=500)
                    
                    for j, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:  # Only include substantial chunks
                            documents.append(chunk.strip())
                            metadatas.append({
                                "section": title,
                                "chunk_id": j,
                                "source": "medical_knowledge_base"
                            })
                            ids.append(f"{title.lower().replace(' ', '_')}_{j}")
            
            # Add documents to collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} medical knowledge chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")

    def _split_into_chunks(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into smaller chunks while preserving context"""
        sentences = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + "\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    async def retrieve_relevant_knowledge(self, features: Dict[str, float], n_results: int = 3) -> str:
        """
        Retrieve relevant medical knowledge based on tumor features
        
        Args:
            features: Dictionary of tumor measurements
            n_results: Number of relevant chunks to retrieve
        
        Returns:
            Concatenated relevant medical knowledge
        """
        try:
            # Create query from features
            query = self._create_feature_query(features)
            
            # Query the vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results["documents"] and len(results["documents"]) > 0:
                # Combine retrieved documents
                relevant_knowledge = "\n\n".join(results["documents"][0])
                logger.info(f"Retrieved {len(results['documents'][0])} relevant knowledge chunks")
                return relevant_knowledge
            else:
                logger.warning("No relevant knowledge found")
                return "No specific medical knowledge retrieved for these features."
                
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return "Error retrieving medical knowledge."

    def _create_feature_query(self, features: Dict[str, float]) -> str:
        """Create a search query based on tumor features"""
        query_parts = []
        
        # Analyze key features and create descriptive query
        if features.get("concave_points_worst", 0) > 0.15:
            query_parts.append("concave points malignant indicators")
        if features.get("radius_worst", 0) > 20:
            query_parts.append("large tumor radius characteristics")
        if features.get("area_worst", 0) > 1500:
            query_parts.append("large tumor area malignancy")
        if features.get("concavity_worst", 0) > 0.3:
            query_parts.append("high concavity malignant features")
        if features.get("texture_worst", 0) > 25:
            query_parts.append("irregular texture patterns")
        
        # Add general terms
        query_parts.extend([
            "tumor characteristics",
            "breast cancer features",
            "diagnostic patterns"
        ])
        
        return " ".join(query_parts[:5])  # Limit query length

    async def add_knowledge(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Add new medical knowledge to the database"""
        try:
            # Generate embedding and add to collection
            doc_id = f"custom_{len(self.collection.get()['ids'])}"
            
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added new knowledge with ID: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the knowledge collection"""
        try:
            collection_data = self.collection.get()
            return {
                "total_documents": len(collection_data["ids"]),
                "collection_name": self.collection_name,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_db_path": self.vector_db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}

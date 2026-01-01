import os
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv
import uuid


# Load environment variables
load_dotenv()

# Get Qdrant configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

class QdrantManager:
    """
    Manager class for handling Qdrant vector database operations
    """
    
    def __init__(self, collection_name: str = "textbook_content_embeddings"):
        """
        Initialize the Qdrant client and collection
        """
        if not QDRANT_URL:
            print("Warning: QDRANT_URL is not set")
        if not QDRANT_API_KEY:
            print("Warning: QDRANT_API_KEY is not set")
            
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=QDRANT_URL or "http://localhost:6333",
            api_key=QDRANT_API_KEY or "dummy",
            prefer_grpc=False
        )
        
        # We'll created the collection only if we have proper credentials
        if QDRANT_URL and QDRANT_API_KEY:
            try:
                self._create_collection_if_not_exists()
            except Exception as e:
                print(f"Warning: Could not check/create Qdrant collection: {e}")
    
    def _create_collection_if_not_exists(self):
        """
        Create the collection if it doesn't already exist
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
        except:
            # Collection doesn't exist, create it
            print(f"Creating collection '{self.collection_name}'...")
            
            # Default vector size for GEMINI embeddings (this might need adjustment based on actual model)
            # GEMINI embeddings typically have different dimensions, so we'll use a placeholder
            # In practice, you would determine this based on the specific GEMINI model used
            vector_size = 768  # Placeholder - this should match your actual embedding dimension
            
            if self.client.collection_exists(collection_name=self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{self.collection_name}' created successfully")
    
    def add_embeddings(self, points: List[Dict[str, Any]]) -> bool:
        """
        Add embeddings to the collection
        Each point should have: id, vector, payload
        """
        try:
            # Prepare points for Qdrant
            qdrant_points = []
            for point in points:
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]  # Contains metadata like module_id, chapter_id, etc.
                    )
                )
            
            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )
            
            return True
        except Exception as e:
            print(f"Error adding embeddings to Qdrant: {e}")
            return False
    
    def search_similar(self, query_vector: List[float], top_k: int = 5, 
                      filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in the collection
        """
        try:
            # Prepare filters if provided
            qdrant_filters = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                
                if filter_conditions:
                    qdrant_filters = models.Filter(
                        must=filter_conditions
                    )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filters
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "payload": hit.payload,
                    "score": hit.score
                })
            
            return results
        except Exception as e:
            print(f"Error searching in Qdrant: {e}")
            return []
    
    def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific embedding by its ID
        """
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[embedding_id]
            )
            
            if records:
                record = records[0]
                return {
                    "id": record.id,
                    "vector": record.vector,
                    "payload": record.payload
                }
            return None
        except Exception as e:
            print(f"Error retrieving embedding from Qdrant: {e}")
            return None
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding by its ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[embedding_id]
            )
            return True
        except Exception as e:
            print(f"Error deleting embedding from Qdrant: {e}")
            return False
    
    def get_collection_info(self):
        """
        Get information about the collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": collection_info.config.params.vectors_count,
                "vector_size": collection_info.config.params.vector_size,
                "distance": collection_info.config.params.distance
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None


# Global instance of QdrantManager (initially None)
_qdrant_manager_instance = None

def get_qdrant_manager() -> QdrantManager:
    """
    Get (and initialize if needed) the global QdrantManager instance
    """
    global _qdrant_manager_instance
    if _qdrant_manager_instance is None:
        _qdrant_manager_instance = QdrantManager()
    return _qdrant_manager_instance


if __name__ == "__main__":
    # Test Qdrant connection
    print("Testing Qdrant connection...")
    try:
        qm = QdrantManager()
        info = qm.get_collection_info()
        if info:
            print(f"Qdrant connection successful! Collection info: {info}")
        else:
            print("Could not retrieve collection info")
    except Exception as e:
        print(f"Qdrant connection failed: {e}")
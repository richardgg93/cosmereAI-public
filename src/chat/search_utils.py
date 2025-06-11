import os
from typing import List, Dict
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient



def search_knowledgebase_single(search_client, embedding_model, search_query: str) -> List[Dict]:
    """
    Searches the knowledge base for relevant information related to a given query.

    Args:
        search_client: An instance of the search client.
        embedding_model: Embedding model.
        search_query: The query to search for in the knowledge base.

    Returns:
        list: A list of dictionaries containing relevant search results.
            Each dictionary includes the following keys:
                - 'id': The ID of the search result.
                - 'score': The relevance score of the search result.
                - 'source': The source of the information (e.g., validation name, filename).
                - 'content': The content of the search result.
    """
    # Define the vectorized query for searching similar documents
    vector_query = VectorizedQuery(vector=embedding_model.predict([search_query])[0],
                              k_nearest_neighbors=5, 
                              fields="vector") 

    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["id", "document", "path", "content"],
        top=5
    )   

    final_results = [
        {
            "id": result["id"],
            "score": result["@search.score"],
            "content": result["content"],
        }
        for result in results
    ]

    return final_results
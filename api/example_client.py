"""
api/example_client.py
----------------------
Example client demonstrating how to use the Scientific Knowledge Graph API.

This script shows how to make requests to all 5 query endpoints and handle responses.

Usage:
    python api/example_client.py
"""

import requests
import json
from typing import Dict, Any


class KnowledgeGraphClient:
    """
    Client for the Scientific Knowledge Graph API.
    
    This class provides convenient methods for querying the knowledge graph
    through the REST API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to an API endpoint.
        
        Args:
            endpoint: API endpoint path (e.g., "/query/cross-study-associations")
            data: Request data dictionary
        
        Returns:
            Response JSON as dictionary
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise
    
    def query_cross_study_associations(
        self,
        disease: str,
        study_type: str = "RCT",
        min_papers: int = 3,
        confidence_threshold: float = 0.7,
        require_open_data: bool = True
    ) -> Dict[str, Any]:
        """
        Query cross-study disease-microbiome associations.
        
        Args:
            disease: Disease entity name
            study_type: Type of study ("RCT", "observational", "meta_analysis", "any")
            min_papers: Minimum number of papers required
            confidence_threshold: Minimum confidence score (0.0-1.0)
            require_open_data: Only include papers with open data
        
        Returns:
            Query response dictionary
        """
        data = {
            "disease": disease,
            "study_type": study_type,
            "min_papers": min_papers,
            "confidence_threshold": confidence_threshold,
            "require_open_data": require_open_data
        }
        
        return self._make_request("/query/cross-study-associations", data)
    
    def query_intervention_evidence(
        self,
        intervention_types: list,
        min_sample_size: int = 50,
        evidence_strength: str = "strong"
    ) -> Dict[str, Any]:
        """
        Query intervention effectiveness evidence.
        
        Args:
            intervention_types: List of intervention types
            min_sample_size: Minimum total sample size
            evidence_strength: Minimum evidence strength ("strong", "moderate", "weak", "any")
        
        Returns:
            Query response dictionary
        """
        data = {
            "intervention_types": intervention_types,
            "min_sample_size": min_sample_size,
            "evidence_strength": evidence_strength
        }
        
        return self._make_request("/query/intervention-evidence", data)
    
    def query_methodology_landscape(
        self,
        year_start: int,
        year_end: int,
        sequencing_methods: list,
        require_deposited_data: bool = True
    ) -> Dict[str, Any]:
        """
        Query methodology landscape and data availability.
        
        Args:
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            sequencing_methods: List of sequencing methods
            require_deposited_data: Only include papers with deposited data
        
        Returns:
            Query response dictionary
        """
        data = {
            "year_start": year_start,
            "year_end": year_end,
            "sequencing_methods": sequencing_methods,
            "require_deposited_data": require_deposited_data
        }
        
        return self._make_request("/query/methodology-landscape", data)
    
    def query_top_associations(
        self,
        disease: str,
        top_n: int = 10,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query top taxa by evidence quality.
        
        Args:
            disease: Disease entity name
            top_n: Maximum number of taxa to return
            min_confidence: Minimum confidence score (0.0-1.0)
        
        Returns:
            Query response dictionary
        """
        data = {
            "disease": disease,
            "top_n": top_n,
            "min_confidence": min_confidence
        }
        
        return self._make_request("/query/top-associations", data)
    
    def query_conflicting_evidence(
        self,
        disease: str,
        min_papers_per_direction: int = 2
    ) -> Dict[str, Any]:
        """
        Query taxa with conflicting evidence.
        
        Args:
            disease: Disease entity name
            min_papers_per_direction: Minimum papers required for each direction
        
        Returns:
            Query response dictionary
        """
        data = {
            "disease": disease,
            "min_papers_per_direction": min_papers_per_direction
        }
        
        return self._make_request("/query/conflicting-evidence", data)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health and Neo4j connectivity.
        
        Returns:
            Health status dictionary
        """
        url = f"{self.base_url}/health"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    
    def invalidate_cache(self) -> Dict[str, Any]:
        """
        Invalidate all cached query results.
        
        Returns:
            Cache invalidation result dictionary
        """
        url = f"{self.base_url}/cache/invalidate"
        response = requests.post(url, timeout=5)
        response.raise_for_status()
        return response.json()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        url = f"{self.base_url}/cache/stats"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()


def print_response(title: str, response: Dict[str, Any]):
    """
    Pretty print a query response.
    
    Args:
        title: Title to display
        response: Response dictionary
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    
    if response.get("success"):
        query_result = response.get("query_result", {})
        print(f"Query: {query_result.get('query_description')}")
        print(f"Results: {query_result.get('result_count')}")
        print(f"Execution time: {query_result.get('execution_time_ms'):.2f}ms")
        
        if query_result.get("result_count", 0) > 0:
            print("\nSample results:")
            results = query_result.get("results", [])
            for i, result in enumerate(results[:3], 1):  # Show first 3 results
                print(f"\n  Result {i}:")
                for key, value in result.items():
                    if isinstance(value, (list, dict)):
                        print(f"    {key}: {json.dumps(value, indent=6)}")
                    else:
                        print(f"    {key}: {value}")
        else:
            print("\nNo results found.")
    else:
        print(f"Error: {response.get('error')}")


def main():
    """
    Main function demonstrating API usage.
    """
    # Initialize client
    client = KnowledgeGraphClient(base_url="http://localhost:8000")
    
    print("Scientific Knowledge Graph API - Example Client")
    print("=" * 80)
    
    # Check API health
    try:
        health = client.health_check()
        print(f"\nAPI Status: {health.get('status')}")
        print(f"Neo4j: {health.get('neo4j')}")
        print(f"Cache: {health.get('cache', 'N/A')}")
    except Exception as e:
        print(f"\nError: API is not available. Make sure the server is running.")
        print(f"Start the server with: uvicorn api.query_api:app --reload")
        return
    
    # Example 1: Cross-study associations
    print("\n\nExample 1: Cross-Study Associations")
    print("-" * 80)
    try:
        response = client.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7,
            require_open_data=True
        )
        print_response("Cross-Study Associations for Type 2 Diabetes", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Intervention evidence
    print("\n\nExample 2: Intervention Evidence")
    print("-" * 80)
    try:
        response = client.query_intervention_evidence(
            intervention_types=["probiotic", "FMT"],
            min_sample_size=50,
            evidence_strength="strong"
        )
        print_response("Intervention Evidence for Probiotics and FMT", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Methodology landscape
    print("\n\nExample 3: Methodology Landscape")
    print("-" * 80)
    try:
        response = client.query_methodology_landscape(
            year_start=2020,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
            require_deposited_data=True
        )
        print_response("Methodology Landscape (2020-2024)", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Top associations
    print("\n\nExample 4: Top Associations")
    print("-" * 80)
    try:
        response = client.query_top_associations(
            disease="IBD",
            top_n=10,
            min_confidence=0.7
        )
        print_response("Top 10 Taxa Associated with IBD", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 5: Conflicting evidence
    print("\n\nExample 5: Conflicting Evidence")
    print("-" * 80)
    try:
        response = client.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=2
        )
        print_response("Conflicting Evidence for Crohn's Disease", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Cache statistics
    print("\n\nCache Statistics")
    print("-" * 80)
    try:
        stats = client.get_cache_stats()
        if stats.get("cache_enabled"):
            print(f"Cache enabled: Yes")
            cache_stats = stats.get("stats", {})
            print(f"Cache stats: {json.dumps(cache_stats, indent=2)}")
        else:
            print("Cache enabled: No")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

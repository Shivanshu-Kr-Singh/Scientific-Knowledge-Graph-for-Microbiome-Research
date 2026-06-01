"""
api/query_complexity_limiter.py
--------------------------------
Query complexity limits for the query API.

This module implements limits on query complexity to prevent expensive
operations that could degrade system performance.

**Validates: Requirement 18.3**

Features:
- Max result count limit (1000 per query)
- Query depth limiting for graph traversals
- Validation before query execution
"""

from typing import Any, Dict, Optional
from loguru import logger


class QueryComplexityLimiter:
    """
    Enforces query complexity limits.
    
    Prevents expensive queries that could impact system performance.
    
    **Validates: Requirement 18.3**
    """
    
    # Maximum number of results per query (Requirement 18.3)
    MAX_RESULT_COUNT = 1000
    
    # Maximum graph traversal depth (prevents expensive path queries)
    MAX_QUERY_DEPTH = 5
    
    def __init__(
        self,
        max_result_count: int = MAX_RESULT_COUNT,
        max_query_depth: int = MAX_QUERY_DEPTH
    ):
        """
        Initialize query complexity limiter.
        
        Args:
            max_result_count: Maximum results allowed per query
            max_query_depth: Maximum graph traversal depth
        """
        self.max_result_count = max_result_count
        self.max_query_depth = max_query_depth
        
        logger.info(
            f"Query complexity limiter initialized: "
            f"max_result_count={max_result_count}, "
            f"max_query_depth={max_query_depth}"
        )
    
    def validate_result_count_limit(self, requested_count: Optional[int]) -> int:
        """
        Validate and cap the requested result count.
        
        Args:
            requested_count: Number of results requested (e.g., top_n parameter)
        
        Returns:
            Capped result count (min of requested and max allowed)
        """
        if requested_count is None:
            return self.max_result_count
        
        if requested_count > self.max_result_count:
            logger.warning(
                f"Requested result count {requested_count} exceeds maximum "
                f"{self.max_result_count}, capping to maximum"
            )
            return self.max_result_count
        
        return requested_count
    
    def apply_result_limit_to_cypher(self, cypher_query: str) -> str:
        """
        Add LIMIT clause to Cypher query if not present.
        
        This ensures all queries have a maximum result count, preventing
        unbounded result sets.
        
        Args:
            cypher_query: Original Cypher query
        
        Returns:
            Modified query with LIMIT clause
        """
        # Check if query already has a LIMIT clause
        query_upper = cypher_query.upper()
        
        if "LIMIT" in query_upper:
            # Query already has a limit, don't modify
            logger.debug("Query already has LIMIT clause, not modifying")
            return cypher_query
        
        # Add LIMIT clause
        limited_query = f"{cypher_query.rstrip()} LIMIT {self.max_result_count}"
        
        logger.debug(
            f"Added LIMIT {self.max_result_count} to query to prevent "
            f"unbounded results"
        )
        
        return limited_query
    
    def validate_query_depth(self, query_params: Dict[str, Any]) -> None:
        """
        Validate query parameters don't request excessive graph depth.
        
        This prevents expensive graph traversals that could timeout or
        consume excessive resources.
        
        Args:
            query_params: Query parameters dictionary
        
        Raises:
            ValueError: If query depth exceeds maximum
        """
        # Check for parameters that might indicate deep traversals
        # For now, we don't have explicit depth parameters in our queries,
        # but this provides a hook for future validation
        
        # Example: if we had a "max_hops" parameter
        if "max_hops" in query_params:
            max_hops = query_params["max_hops"]
            if max_hops > self.max_query_depth:
                raise ValueError(
                    f"Query depth {max_hops} exceeds maximum allowed "
                    f"depth {self.max_query_depth}"
                )
        
        logger.debug("Query depth validation passed")
    
    def limit_result_set(self, results: list, query_name: str) -> list:
        """
        Enforce result count limit on query results.
        
        This is a safety net in case the database returns more results
        than expected.
        
        Args:
            results: List of query results
            query_name: Name of the query (for logging)
        
        Returns:
            Limited result list
        """
        if len(results) > self.max_result_count:
            logger.warning(
                f"Query '{query_name}' returned {len(results)} results, "
                f"limiting to {self.max_result_count}"
            )
            return results[:self.max_result_count]
        
        return results
    
    def get_limits(self) -> Dict[str, int]:
        """
        Get current complexity limits.
        
        Returns:
            Dictionary with limit values
        """
        return {
            "max_result_count": self.max_result_count,
            "max_query_depth": self.max_query_depth
        }


# Global query complexity limiter instance
query_complexity_limiter = QueryComplexityLimiter()

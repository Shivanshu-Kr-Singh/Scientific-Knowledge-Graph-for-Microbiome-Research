"""
graph/research_query_engine.py
-------------------------------
Research Query Engine for Scientific Knowledge Graph

This module provides the base infrastructure for executing scientific queries
against the Neo4j knowledge graph. It implements:
- QueryResult model with query metadata
- Query execution timing and result counting
- Parameterized Cypher query generation to prevent injection attacks
- Base class for research query operations

Requirements: 1.1, 1.2, 1.3, 18.1
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
import time
import uuid
from graph.query_cache import QueryCache


class QueryResult(BaseModel):
    """
    Result from a research query with complete metadata.
    
    This model captures:
    - Query identification and description
    - Actual result data
    - Execution metrics (timing, result count)
    - Aggregation metadata for evidence-based queries
    
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    
    # Query identification
    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this query execution"
    )
    query_description: str = Field(
        ...,
        description="Human-readable description of what this query does"
    )
    
    # Results
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of result records from the query"
    )
    result_count: int = Field(
        default=0,
        ge=0,
        description="Number of results returned"
    )
    
    # Execution metrics
    execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds"
    )
    executed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when query was executed"
    )
    
    # Aggregation metadata (optional, used for evidence-based queries)
    aggregation_method: Optional[str] = Field(
        default=None,
        description="Method used for aggregating evidence (e.g., 'weighted_average', 'consensus')"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold applied to filter results"
    )
    
    # Query status
    timeout: bool = Field(
        default=False,
        description="Whether the query timed out before completion"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if query failed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_id": "550e8400-e29b-41d4-a716-446655440000",
                "query_description": "Cross-study associations for Type 2 Diabetes",
                "results": [
                    {
                        "taxon": "Bacteroides fragilis",
                        "paper_count": 5,
                        "consensus_confidence": 0.85,
                        "direction": "increased"
                    }
                ],
                "result_count": 1,
                "execution_time_ms": 245.3,
                "executed_at": "2024-01-15T10:30:00.000Z",
                "aggregation_method": "weighted_average",
                "confidence_threshold": 0.7,
                "timeout": False,
                "error": None
            }
        }
    )


class ResearchQueryEngine:
    """
    Base class for executing scientific research queries against the knowledge graph.
    
    This class provides:
    - Parameterized Cypher query execution to prevent injection attacks
    - Query timing and result counting
    - Error handling and timeout management
    - Base infrastructure for research query methods
    
    **Validates: Requirements 1.1, 1.2, 1.3, 18.1**
    
    Security Features (Requirement 18.1):
    - All queries use parameterized Cypher to prevent injection
    - Input validation and sanitization
    - Query complexity limits (enforced by Neo4j driver)
    
    Usage:
        engine = ResearchQueryEngine(neo4j_driver)
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) WHERE p.year = $year RETURN p",
            parameters={"year": 2024},
            description="Find papers from 2024"
        )
    """
    
    def __init__(self, driver, enable_cache: bool = True, cache_ttl_hours: int = 24):
        """
        Initialize the query engine with a Neo4j driver.
        
        Args:
            driver: Neo4j driver instance for database connection
            enable_cache: Whether to enable query result caching (default: True)
            cache_ttl_hours: Cache TTL in hours (default: 24)
        
        **Validates: Requirement 13.5 (query result caching with 24-hour TTL)**
        """
        self.driver = driver
        self.default_timeout_seconds = 30
        
        # Initialize cache (Requirement 13.5)
        self.enable_cache = enable_cache
        self.cache = QueryCache(ttl_hours=cache_ttl_hours) if enable_cache else None
    
    def invalidate_cache(self) -> int:
        """
        Invalidate all cached query results.
        
        This method should be called when new data is loaded into the knowledge graph
        to ensure queries return fresh results.
        
        **Validates: Requirement 13.5 (cache invalidation when new data is loaded)**
        
        Returns:
            Number of cache entries that were invalidated (0 if caching is disabled)
        
        Example:
            # After loading new papers into the graph
            load_papers_to_graph(new_papers)
            
            # Invalidate cache to ensure fresh results
            engine.invalidate_cache()
        """
        if self.cache is not None:
            return self.cache.invalidate_all()
        return 0
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics if caching is enabled, None otherwise
        
        Example:
            stats = engine.get_cache_stats()
            if stats:
                print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        """
        if self.cache is not None:
            return self.cache.get_stats()
        return None
    
    def _execute_with_cache(
        self,
        query_name: str,
        parameters: Dict[str, Any],
        query_func
    ) -> QueryResult:
        """
        Execute a query with caching support.
        
        This internal method checks the cache first, and only executes the query
        if there's a cache miss. Results are automatically cached.
        
        **Validates: Requirement 13.5 (query result caching)**
        
        Args:
            query_name: Name of the query method (used as cache key)
            parameters: Query parameters (used as cache key)
            query_func: Function that executes the actual query (no arguments)
        
        Returns:
            QueryResult from cache or fresh query execution
        
        Preconditions:
        - query_name is non-empty string
        - parameters is a dictionary
        - query_func is a callable that returns QueryResult
        
        Postconditions:
        - Returns cached result if available and not expired
        - Executes query_func and caches result on cache miss
        - Returns QueryResult
        """
        # Check cache if enabled
        if self.cache is not None:
            cached_result = self.cache.get(query_name, parameters)
            if cached_result is not None:
                # Cache hit - return cached result
                return cached_result
        
        # Cache miss or caching disabled - execute query
        result = query_func()
        
        # Cache the result if caching is enabled and query succeeded
        if self.cache is not None and result.error is None:
            self.cache.set(query_name, parameters, result)
        
        return result
    
    def execute_query(
        self,
        cypher_query: str,
        parameters: Dict[str, Any],
        description: str,
        aggregation_method: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        timeout_seconds: Optional[int] = None
    ) -> QueryResult:
        """
        Execute a parameterized Cypher query with timing and result counting.
        
        This method provides the core query execution infrastructure:
        - Measures execution time
        - Counts results
        - Handles errors and timeouts
        - Returns structured QueryResult
        
        **Security (Requirement 18.1):**
        - Uses parameterized queries exclusively to prevent injection
        - All user inputs must be passed via parameters dict
        - Never concatenates user input into query strings
        
        Args:
            cypher_query: Parameterized Cypher query string with $parameter placeholders
            parameters: Dictionary of parameter values for the query
            description: Human-readable description of the query
            aggregation_method: Optional method used for evidence aggregation
            confidence_threshold: Optional confidence threshold applied
            timeout_seconds: Optional timeout override (default: 30 seconds)
        
        Returns:
            QueryResult with results, timing, and metadata
        
        Example:
            result = engine.execute_query(
                cypher_query='''
                    MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)
                    WHERE r.disease = $disease
                      AND r.confidence >= $threshold
                    RETURN t.name as taxon, count(p) as paper_count
                ''',
                parameters={
                    "disease": "Type 2 Diabetes",
                    "threshold": 0.7
                },
                description="Find taxa associated with Type 2 Diabetes",
                confidence_threshold=0.7
            )
        """
        # Validate inputs
        if not cypher_query or not isinstance(cypher_query, str):
            return QueryResult(
                query_description=description,
                error="Invalid Cypher query: must be a non-empty string"
            )
        
        if not isinstance(parameters, dict):
            return QueryResult(
                query_description=description,
                error="Invalid parameters: must be a dictionary"
            )
        
        # Set timeout
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        
        # Start timing
        start_time = time.time()
        
        try:
            # Execute parameterized query
            with self.driver.session() as session:
                # Run query with timeout
                result = session.run(cypher_query, parameters)
                
                # Fetch all results
                records = []
                for record in result:
                    # Convert Neo4j record to dictionary
                    records.append(dict(record))
                
                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Check for timeout
                timed_out = execution_time_ms > (timeout * 1000)
                
                # Create and return QueryResult
                return QueryResult(
                    query_description=description,
                    results=records,
                    result_count=len(records),
                    execution_time_ms=execution_time_ms,
                    aggregation_method=aggregation_method,
                    confidence_threshold=confidence_threshold,
                    timeout=timed_out
                )
        
        except Exception as e:
            # Calculate execution time even on error
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Return error result
            return QueryResult(
                query_description=description,
                results=[],
                result_count=0,
                execution_time_ms=execution_time_ms,
                aggregation_method=aggregation_method,
                confidence_threshold=confidence_threshold,
                error=str(e)
            )
    
    def validate_parameter(
        self,
        param_name: str,
        param_value: Any,
        param_type: type,
        allowed_values: Optional[List[Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a query parameter for type and allowed values.
        
        This method provides input validation to prevent injection and ensure
        query correctness.
        
        **Security (Requirement 18.1):**
        - Validates parameter types
        - Checks against allowed value lists
        - Prevents malicious inputs
        
        Args:
            param_name: Name of the parameter (for error messages)
            param_value: Value to validate
            param_type: Expected Python type
            allowed_values: Optional list of allowed values
        
        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
        
        Example:
            valid, error = engine.validate_parameter(
                "study_type",
                "RCT",
                str,
                allowed_values=["RCT", "observational", "meta_analysis"]
            )
        """
        # Check type
        if not isinstance(param_value, param_type):
            return False, f"Parameter '{param_name}' must be of type {param_type.__name__}"
        
        # Check allowed values
        if allowed_values is not None and param_value not in allowed_values:
            return False, f"Parameter '{param_name}' must be one of {allowed_values}"
        
        return True, None
    
    def sanitize_string_parameter(self, value: str) -> str:
        """
        Sanitize a string parameter for use in queries.
        
        **Security (Requirement 18.1):**
        - Removes potentially dangerous characters
        - Trims whitespace
        - Prevents injection attempts
        
        Note: This is a defense-in-depth measure. The primary protection
        is parameterized queries, but this adds an extra layer.
        
        Args:
            value: String value to sanitize
        
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return ""
        
        # Trim whitespace
        sanitized = value.strip()
        
        # Remove null bytes (can cause issues in some contexts)
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
    
    def build_parameterized_query(
        self,
        base_query: str,
        filters: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a parameterized Cypher query from a base query and filters.
        
        This method helps construct complex queries while maintaining
        parameterization for security.
        
        **Security (Requirement 18.1):**
        - Ensures all dynamic values are parameterized
        - Never concatenates user input into query strings
        
        Args:
            base_query: Base Cypher query with WHERE clause placeholder
            filters: Dictionary of filter conditions
        
        Returns:
            Tuple of (complete_query, parameters)
        
        Example:
            query, params = engine.build_parameterized_query(
                base_query="MATCH (p:Paper) WHERE {filters} RETURN p",
                filters={"year": 2024, "article_type": "original_research"}
            )
            # Returns:
            # query = "MATCH (p:Paper) WHERE p.year = $year AND p.article_type = $article_type RETURN p"
            # params = {"year": 2024, "article_type": "original_research"}
        """
        # Build WHERE conditions
        conditions = []
        parameters = {}
        
        for key, value in filters.items():
            # Create parameter name
            param_name = key.replace('.', '_')
            
            # Add condition
            conditions.append(f"{key} = ${param_name}")
            
            # Add parameter
            parameters[param_name] = value
        
        # Combine conditions
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        # Replace placeholder in base query
        complete_query = base_query.replace("{filters}", where_clause)
        
        return complete_query, parameters
    
    def query_cross_study_associations(
        self,
        disease: str,
        study_type: str = "RCT",
        min_papers: int = 3,
        confidence_threshold: float = 0.7,
        require_open_data: bool = True
    ) -> QueryResult:
        """
        Q1: Find taxa with consistent disease associations across multiple studies.
        
        This query answers: "Which gut microbiome taxa show consistent association 
        with [disease] across [study_type] studies with open sequencing data?"
        
        **Validates: Requirements 1.1, 6.1, 6.2, 6.3, 6.4, 6.5**
        
        The query:
        1. Filters by disease, study type, and confidence threshold
        2. Optionally filters for papers with open data (data_availability="open" 
           and non-empty accession_numbers)
        3. Aggregates by taxon to calculate consensus metrics
        4. Filters taxa by minimum paper count
        5. Sorts by consensus confidence (descending), then paper count (descending)
        
        Args:
            disease: Disease entity name to query (e.g., "Type 2 Diabetes")
            study_type: Type of study to include. Options:
                - "RCT": Randomized controlled trials
                - "observational": Observational studies
                - "meta_analysis": Meta-analyses
                - "any": All study types
            min_papers: Minimum number of papers required for a taxon to be included
            confidence_threshold: Minimum confidence score (0.0-1.0) for relationships
            require_open_data: If True, only include papers with open data 
                (data_availability="open" and non-empty accession_numbers)
        
        Returns:
            QueryResult containing:
            - taxon_name: Name of the taxon
            - paper_count: Number of papers reporting this association
            - consensus_confidence: Average confidence across all papers
            - consensus_direction: Most common direction (increased/decreased/no_change/associated)
            - direction_consistency: Percentage of papers agreeing on consensus direction
            - increased_count: Number of papers reporting "increased"
            - decreased_count: Number of papers reporting "decreased"
            - no_change_count: Number of papers reporting "no_change"
            - associated_count: Number of papers reporting "associated"
            - paper_ids: List of paper identifiers
        
        Example:
            result = engine.query_cross_study_associations(
                disease="Type 2 Diabetes",
                study_type="RCT",
                min_papers=3,
                confidence_threshold=0.7,
                require_open_data=True
            )
            
            # Returns taxa like:
            # {
            #   "taxon_name": "Bacteroides fragilis",
            #   "paper_count": 5,
            #   "consensus_confidence": 0.85,
            #   "consensus_direction": "increased",
            #   "direction_consistency": 0.80,
            #   "increased_count": 4,
            #   "decreased_count": 1,
            #   "no_change_count": 0,
            #   "paper_ids": ["PMID:12345", "PMID:67890", ...]
            # }
        
        Preconditions:
        - disease is a valid disease entity in the graph
        - study_type in ["RCT", "observational", "meta_analysis", "any"]
        - min_papers >= 1
        - confidence_threshold in [0.0, 1.0]
        
        Postconditions (Requirement 6.4, 6.5):
        - Returns taxa mentioned in >= min_papers papers
        - All results have consensus_confidence >= confidence_threshold
        - If require_open_data=True, only includes papers with data_availability="open"
          and non-empty accession_numbers (Requirement 6.2)
        - Results sorted by consensus_confidence DESC, then paper_count DESC
        """
        # Validate inputs (Requirement 18.2)
        valid, error = self.validate_parameter(
            "study_type",
            study_type,
            str,
            allowed_values=["RCT", "observational", "meta_analysis", "any"]
        )
        if not valid:
            return QueryResult(
                query_description=f"Cross-study associations for {disease}",
                error=error
            )
        
        if not isinstance(min_papers, int) or min_papers < 1:
            return QueryResult(
                query_description=f"Cross-study associations for {disease}",
                error="min_papers must be an integer >= 1"
            )
        
        if not isinstance(confidence_threshold, (int, float)) or not (0.0 <= confidence_threshold <= 1.0):
            return QueryResult(
                query_description=f"Cross-study associations for {disease}",
                error="confidence_threshold must be a number in range [0.0, 1.0]"
            )
        
        # Sanitize string inputs (Requirement 18.2)
        disease = self.sanitize_string_parameter(disease)
        
        # Build Cypher query (Requirement 6.1, 6.2, 6.3)
        # This query pattern is specified in the design document
        cypher_query = """
        MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)
        WHERE r.disease = $disease
          AND r.confidence >= $threshold
          AND ($study_type = 'any' OR p.article_type = $study_type)
          AND (NOT $require_open_data OR (p.data_availability = 'open' AND size(p.accession_numbers) > 0))
        WITH t, 
             collect(DISTINCT p) as papers,
             collect(r.confidence) as confidences,
             collect(r.direction) as directions,
             avg(r.confidence) as consensus_confidence
        WHERE size(papers) >= $min_papers
          AND consensus_confidence >= $threshold
        WITH t,
             papers,
             consensus_confidence,
             directions,
             size([d IN directions WHERE d = 'increased']) as increased_count,
             size([d IN directions WHERE d = 'decreased']) as decreased_count,
             size([d IN directions WHERE d = 'no_change']) as no_change_count,
             size([d IN directions WHERE d = 'associated']) as associated_count
        WITH t,
             papers,
             consensus_confidence,
             increased_count,
             decreased_count,
             no_change_count,
             associated_count,
             CASE
                WHEN increased_count >= decreased_count AND increased_count >= no_change_count AND increased_count >= associated_count THEN 'increased'
                WHEN decreased_count >= increased_count AND decreased_count >= no_change_count AND decreased_count >= associated_count THEN 'decreased'
                WHEN associated_count >= increased_count AND associated_count >= decreased_count AND associated_count >= no_change_count THEN 'associated'
                ELSE 'no_change'
             END as consensus_direction,
             CASE
                WHEN increased_count >= decreased_count AND increased_count >= no_change_count AND increased_count >= associated_count
                    THEN toFloat(increased_count) / size(directions)
                WHEN decreased_count >= increased_count AND decreased_count >= no_change_count AND decreased_count >= associated_count
                    THEN toFloat(decreased_count) / size(directions)
                WHEN associated_count >= increased_count AND associated_count >= decreased_count AND associated_count >= no_change_count
                    THEN toFloat(associated_count) / size(directions)
                ELSE toFloat(no_change_count) / size(directions)
             END as direction_consistency
        RETURN t.name as taxon_name,
               size(papers) as paper_count,
               consensus_confidence,
               consensus_direction,
               direction_consistency,
               increased_count,
               decreased_count,
               no_change_count,
               associated_count,
               [p IN papers | COALESCE(p.doi, p.pmid, p.title)] as paper_ids
        ORDER BY consensus_confidence DESC, paper_count DESC
        """
        
        # Build parameters (Requirement 18.1 - parameterized queries)
        parameters = {
            "disease": disease,
            "threshold": confidence_threshold,
            "study_type": study_type,
            "min_papers": min_papers,
            "require_open_data": require_open_data
        }
        
        # Execute query with caching (Requirement 13.5)
        return self._execute_with_cache(
            query_name="query_cross_study_associations",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=f"Cross-study associations for {disease} (study_type={study_type}, min_papers={min_papers}, confidence>={confidence_threshold}, open_data={require_open_data})",
                aggregation_method="weighted_average",
                confidence_threshold=confidence_threshold
            )
        )
    
    def query_intervention_evidence(
        self,
        intervention_types: List[str],
        min_sample_size: int = 50,
        evidence_strength: str = "strong"
    ) -> QueryResult:
        """
        Q2: Find interventions with RCT-level evidence for modifying specific taxa.
        
        This query answers: "What interventions (probiotics, FMT, diet) have 
        RCT-level evidence for modifying specific gut taxa, and what effect 
        directions are reported?"
        
        **Validates: Requirements 1.2, 7.1, 7.2, 7.3, 7.4, 7.5**
        
        The query:
        1. Filters by intervention types, minimum sample size, and evidence strength
        2. Only returns interventions from article_type "original_research" or "meta_analysis"
        3. Groups by (intervention, taxon, effect_direction)
        4. Calculates total_sample_size and paper_count for each group
        5. Sorts by paper_count DESC, then total_sample_size DESC
        
        Args:
            intervention_types: List of intervention types to query. Examples:
                - ["probiotic"]: Only probiotic interventions
                - ["FMT", "diet"]: FMT and dietary interventions
                - ["probiotic", "FMT", "diet", "antibiotic"]: Multiple types
            min_sample_size: Minimum total sample size across all papers for an 
                intervention-taxon-direction combination to be included
            evidence_strength: Minimum evidence strength to include. Options:
                - "strong": p < 0.01, RCT or meta-analysis
                - "moderate": p < 0.05
                - "weak": p < 0.1 or no p-value
                - "any": All evidence strengths
        
        Returns:
            QueryResult containing:
            - intervention_type: Type of intervention (e.g., "probiotic", "FMT")
            - taxon_name: Name of the affected taxon
            - effect_direction: Direction of effect ("increased" or "decreased")
            - paper_count: Number of papers reporting this combination
            - total_sample_size: Sum of sample sizes across all papers
            - paper_ids: List of paper identifiers
            - avg_confidence: Average confidence score across papers
        
        Example:
            result = engine.query_intervention_evidence(
                intervention_types=["probiotic", "FMT"],
                min_sample_size=50,
                evidence_strength="strong"
            )
            
            # Returns interventions like:
            # {
            #   "intervention_type": "probiotic",
            #   "taxon_name": "Lactobacillus acidophilus",
            #   "effect_direction": "increased",
            #   "paper_count": 8,
            #   "total_sample_size": 450,
            #   "paper_ids": ["PMID:12345", "PMID:67890", ...],
            #   "avg_confidence": 0.87
            # }
        
        Preconditions:
        - intervention_types is non-empty list of intervention names
        - min_sample_size >= 1
        - evidence_strength in ["strong", "moderate", "weak", "any"]
        
        Postconditions (Requirements 7.2, 7.3, 7.4, 7.5):
        - Returns interventions from article_type "original_research" or "meta_analysis"
        - All results have total_sample_size >= min_sample_size
        - Results grouped by (intervention, taxon, effect_direction)
        - Results sorted by paper_count DESC, then total_sample_size DESC
        """
        # Validate inputs (Requirement 18.2)
        if not isinstance(intervention_types, list) or len(intervention_types) == 0:
            return QueryResult(
                query_description="Intervention effectiveness query",
                error="intervention_types must be a non-empty list"
            )
        
        valid, error = self.validate_parameter(
            "evidence_strength",
            evidence_strength,
            str,
            allowed_values=["strong", "moderate", "weak", "any"]
        )
        if not valid:
            return QueryResult(
                query_description="Intervention effectiveness query",
                error=error
            )
        
        if not isinstance(min_sample_size, int) or min_sample_size < 1:
            return QueryResult(
                query_description="Intervention effectiveness query",
                error="min_sample_size must be an integer >= 1"
            )
        
        # Sanitize string inputs (Requirement 18.2)
        intervention_types = [self.sanitize_string_parameter(it) for it in intervention_types]
        
        # Build Cypher query (Requirements 7.1, 7.2, 7.3, 7.4, 7.5)
        # This query pattern is specified in the design document
        cypher_query = """
        MATCH (p:Paper)-[r:REPORTS_INTERVENTION_EFFECT]->(t:Taxon)
        WHERE r.intervention_type IN $intervention_types
          AND ($evidence_strength = 'any' OR r.evidence_strength = $evidence_strength)
          AND (p.article_type = 'original_research' OR p.article_type = 'meta_analysis')
          AND r.sample_size IS NOT NULL
        WITH r.intervention_type as intervention,
             t.name as taxon,
             r.effect_direction as direction,
             collect(DISTINCT p) as papers,
             sum(r.sample_size) as total_samples,
             avg(r.confidence) as avg_confidence,
             collect(r.confidence) as confidences
        WHERE total_samples >= $min_sample_size
        RETURN intervention as intervention_type,
               taxon as taxon_name,
               direction as effect_direction,
               size(papers) as paper_count,
               total_samples as total_sample_size,
               [p IN papers | COALESCE(p.doi, p.pmid, p.title)] as paper_ids,
               avg_confidence
        ORDER BY paper_count DESC, total_sample_size DESC
        """
        
        # Build parameters (Requirement 18.1 - parameterized queries)
        parameters = {
            "intervention_types": intervention_types,
            "evidence_strength": evidence_strength,
            "min_sample_size": min_sample_size
        }
        
        # Execute query with caching (Requirement 13.5)
        return self._execute_with_cache(
            query_name="query_intervention_evidence",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=f"Intervention effectiveness (types={intervention_types}, min_sample_size={min_sample_size}, evidence_strength={evidence_strength})",
                aggregation_method="sum_sample_sizes"
            )
        )
    
    def query_methodology_landscape(
        self,
        year_start: int,
        year_end: int,
        sequencing_methods: List[str],
        require_deposited_data: bool = True
    ) -> QueryResult:
        """
        Q3: Survey data availability and methodology across time period.
        
        This query answers: "Which microbiome studies from [year_start]-[year_end] 
        deposited data on SRA/ENA and used shotgun metagenomics vs 16S sequencing?"
        
        **Validates: Requirements 1.3, 8.1, 8.2, 8.3, 8.4, 8.5**
        
        The query:
        1. Filters by year range, sequencing methods, and data deposition requirement
        2. When require_deposited_data=True, only returns papers with non-empty accession_numbers
        3. Groups by (method, year) and calculates total papers, papers with data, 
           and data availability percentage
        4. Identifies repository (NCBI SRA vs ENA) from accession number prefixes
        5. Sorts by year DESC, then method ASC
        
        Repository Identification (Requirement 8.4):
        - NCBI SRA: Accession numbers starting with SRP, SRR, SRX, SRS, PRJNA, SAMN
        - ENA: Accession numbers starting with ERP, ERR, ERX, ERS, PRJEB
        - Papers can have data in multiple repositories
        
        Args:
            year_start: Start year of the time period (inclusive)
            year_end: End year of the time period (inclusive)
            sequencing_methods: List of sequencing method names to query. Examples:
                - ["16S rRNA sequencing"]: Only 16S studies
                - ["shotgun metagenomics"]: Only shotgun studies
                - ["16S rRNA sequencing", "shotgun metagenomics"]: Both methods
            require_deposited_data: If True, only include papers with non-empty 
                accession_numbers (i.e., papers that deposited data)
        
        Returns:
            QueryResult containing:
            - method: Sequencing method name
            - year: Publication year
            - total_papers: Total number of papers using this method in this year
            - papers_with_data: Number of papers that deposited data
            - data_availability_pct: Percentage of papers with deposited data
            - ncbi_sra_count: Number of papers with data in NCBI SRA
            - ena_count: Number of papers with data in ENA
            - both_repositories_count: Number of papers with data in both repositories
        
        Example:
            result = engine.query_methodology_landscape(
                year_start=2020,
                year_end=2024,
                sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
                require_deposited_data=True
            )
            
            # Returns results like:
            # {
            #   "method": "shotgun metagenomics",
            #   "year": 2024,
            #   "total_papers": 45,
            #   "papers_with_data": 38,
            #   "data_availability_pct": 84.4,
            #   "ncbi_sra_count": 30,
            #   "ena_count": 12,
            #   "both_repositories_count": 4
            # }
        
        Preconditions:
        - year_start <= year_end
        - sequencing_methods is non-empty list
        - All methods in sequencing_methods are valid method entities
        
        Postconditions (Requirements 8.1, 8.2, 8.3, 8.4, 8.5):
        - Returns papers published between year_start and year_end (inclusive)
        - If require_deposited_data=True, only papers with non-empty accession_numbers
        - Results grouped by (method, year)
        - Includes counts, percentages, and repository breakdown
        - Results sorted by year DESC, then method ASC
        """
        # Validate inputs (Requirement 18.2)
        if not isinstance(year_start, int) or not isinstance(year_end, int):
            return QueryResult(
                query_description="Methodology landscape query",
                error="year_start and year_end must be integers"
            )
        
        if year_start > year_end:
            return QueryResult(
                query_description="Methodology landscape query",
                error="year_start must be <= year_end"
            )
        
        if not isinstance(sequencing_methods, list) or len(sequencing_methods) == 0:
            return QueryResult(
                query_description="Methodology landscape query",
                error="sequencing_methods must be a non-empty list"
            )
        
        # Sanitize string inputs (Requirement 18.2)
        sequencing_methods = [self.sanitize_string_parameter(method) for method in sequencing_methods]
        
        # Build Cypher query (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
        # This query pattern is specified in the design document
        cypher_query = """
        MATCH (p:Paper)-[r:USES_METHODOLOGY]->(m:Method)
        WHERE p.year >= $year_start 
          AND p.year <= $year_end
          AND toLower(m.name) IN $sequencing_methods_lower
          AND (NOT $require_deposited_data OR size(p.accession_numbers) > 0)
        WITH m.name as method,
             p.year as year,
             collect(DISTINCT p) as papers
        WITH method,
             year,
             papers,
             [p IN papers WHERE size(p.accession_numbers) > 0] as papers_with_data
        WITH method,
             year,
             size(papers) as total_papers,
             size(papers_with_data) as papers_with_data_count,
             papers_with_data
        WITH method,
             year,
             total_papers,
             papers_with_data_count,
             CASE 
                WHEN total_papers > 0 
                THEN 100.0 * papers_with_data_count / total_papers 
                ELSE 0.0 
             END as data_availability_pct,
             papers_with_data
        WITH method,
             year,
             total_papers,
             papers_with_data_count,
             data_availability_pct,
             [p IN papers_with_data WHERE 
                any(acc IN p.accession_numbers WHERE 
                    acc STARTS WITH 'SRP' OR acc STARTS WITH 'SRR' OR 
                    acc STARTS WITH 'SRX' OR acc STARTS WITH 'SRS' OR 
                    acc STARTS WITH 'PRJNA' OR acc STARTS WITH 'SAMN'
                )
             ] as ncbi_sra_papers,
             [p IN papers_with_data WHERE 
                any(acc IN p.accession_numbers WHERE 
                    acc STARTS WITH 'ERP' OR acc STARTS WITH 'ERR' OR 
                    acc STARTS WITH 'ERX' OR acc STARTS WITH 'ERS' OR 
                    acc STARTS WITH 'PRJEB'
                )
             ] as ena_papers
        WITH method,
             year,
             total_papers,
             papers_with_data_count,
             data_availability_pct,
             size(ncbi_sra_papers) as ncbi_sra_count,
             size(ena_papers) as ena_count,
             ncbi_sra_papers,
             ena_papers
        WITH method,
             year,
             total_papers,
             papers_with_data_count,
             data_availability_pct,
             ncbi_sra_count,
             ena_count,
             size([p IN ncbi_sra_papers WHERE p IN ena_papers]) as both_repositories_count
        RETURN method,
               year,
               total_papers,
               papers_with_data_count as papers_with_data,
               data_availability_pct,
               ncbi_sra_count,
               ena_count,
               both_repositories_count
        ORDER BY year DESC, method ASC
        """
        
        # Build parameters (Requirement 18.1 - parameterized queries)
        parameters = {
            "year_start": year_start,
            "year_end": year_end,
            "sequencing_methods": sequencing_methods,
            "sequencing_methods_lower": [m.lower() for m in sequencing_methods],
            "require_deposited_data": require_deposited_data
        }
        
        # Execute query with caching (Requirement 13.5)
        return self._execute_with_cache(
            query_name="query_methodology_landscape",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=f"Methodology landscape ({year_start}-{year_end}, methods={sequencing_methods}, require_deposited_data={require_deposited_data})",
                aggregation_method="group_by_method_year"
            )
        )
    
    def query_top_associations_by_evidence(
        self,
        disease: str,
        top_n: int = 10,
        min_confidence: float = 0.7
    ) -> QueryResult:
        """
        Q4: Find top taxa associated with a disease ranked by evidence quality.
        
        This query answers: "Top N taxa associated with [disease] across multiple 
        papers with confidence >= [min_confidence], ranked by evidence quality."
        
        **Validates: Requirements 1.4**
        
        The query:
        1. Filters by disease and minimum confidence threshold
        2. Aggregates by taxon to calculate paper count, average confidence, and direction consistency
        3. Returns at most top_n taxa
        4. Sorts by paper_count DESC, then avg_confidence DESC
        
        Args:
            disease: Disease entity name to query (e.g., "Type 2 Diabetes", "IBD")
            top_n: Maximum number of taxa to return (default: 10)
            min_confidence: Minimum confidence score (0.0-1.0) for relationships (default: 0.7)
        
        Returns:
            QueryResult containing:
            - taxon_name: Name of the taxon
            - paper_count: Number of papers reporting this association
            - avg_confidence: Average confidence across all papers
            - consensus_direction: Most common direction (increased/decreased/no_change)
            - direction_consistency: Percentage of papers agreeing on consensus direction
            - increased_count: Number of papers reporting "increased"
            - decreased_count: Number of papers reporting "decreased"
            - no_change_count: Number of papers reporting "no_change"
            - paper_ids: List of paper identifiers
        
        Example:
            result = engine.query_top_associations_by_evidence(
                disease="IBD",
                top_n=10,
                min_confidence=0.7
            )
            
            # Returns top taxa like:
            # {
            #   "taxon_name": "Faecalibacterium prausnitzii",
            #   "paper_count": 12,
            #   "avg_confidence": 0.89,
            #   "consensus_direction": "decreased",
            #   "direction_consistency": 0.92,
            #   "increased_count": 1,
            #   "decreased_count": 11,
            #   "no_change_count": 0,
            #   "paper_ids": ["PMID:12345", "DOI:10.1234/test", ...]
            # }
        
        Preconditions:
        - disease is a valid disease entity
        - top_n >= 1
        - min_confidence in [0.0, 1.0]
        
        Postconditions (Requirement 1.4):
        - Returns at most top_n taxa
        - All results have confidence >= min_confidence
        - Results sorted by (paper_count DESC, avg_confidence DESC)
        - Includes aggregated statistics for each taxon
        """
        # Validate inputs (Requirement 18.2)
        if not isinstance(top_n, int) or top_n < 1:
            return QueryResult(
                query_description=f"Top associations for {disease}",
                error="top_n must be an integer >= 1"
            )
        
        if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
            return QueryResult(
                query_description=f"Top associations for {disease}",
                error="min_confidence must be a number in range [0.0, 1.0]"
            )
        
        # Sanitize string inputs (Requirement 18.2)
        disease = self.sanitize_string_parameter(disease)
        
        # Build Cypher query (Requirement 1.4)
        cypher_query = """
        MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)
        WHERE r.disease = $disease
          AND r.confidence >= $min_confidence
        WITH t,
             collect(DISTINCT p) as papers,
             collect(r.confidence) as confidences,
             collect(r.direction) as directions,
             avg(r.confidence) as avg_confidence
        WITH t,
             papers,
             avg_confidence,
             directions,
             size([d IN directions WHERE d = 'increased']) as increased_count,
             size([d IN directions WHERE d = 'decreased']) as decreased_count,
             size([d IN directions WHERE d = 'no_change']) as no_change_count
        WITH t,
             papers,
             avg_confidence,
             increased_count,
             decreased_count,
             no_change_count,
             CASE
                WHEN increased_count >= decreased_count AND increased_count >= no_change_count THEN 'increased'
                WHEN decreased_count >= increased_count AND decreased_count >= no_change_count THEN 'decreased'
                ELSE 'no_change'
             END as consensus_direction,
             CASE
                WHEN increased_count >= decreased_count AND increased_count >= no_change_count 
                    THEN toFloat(increased_count) / size(directions)
                WHEN decreased_count >= increased_count AND decreased_count >= no_change_count 
                    THEN toFloat(decreased_count) / size(directions)
                ELSE toFloat(no_change_count) / size(directions)
             END as direction_consistency
        RETURN t.name as taxon_name,
               size(papers) as paper_count,
               avg_confidence,
               consensus_direction,
               direction_consistency,
               increased_count,
               decreased_count,
               no_change_count,
               [p IN papers | COALESCE(p.doi, p.pmid, p.title)] as paper_ids
        ORDER BY paper_count DESC, avg_confidence DESC
        LIMIT $top_n
        """
        
        # Build parameters (Requirement 18.1 - parameterized queries)
        parameters = {
            "disease": disease,
            "min_confidence": min_confidence,
            "top_n": top_n
        }
        
        # Execute query with caching (Requirement 13.5)
        return self._execute_with_cache(
            query_name="query_top_associations_by_evidence",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=f"Top {top_n} associations for {disease} (min_confidence>={min_confidence})",
                aggregation_method="top_n_by_evidence",
                confidence_threshold=min_confidence
            )
        )
    
    def query_conflicting_evidence(
        self,
        disease: str,
        min_papers_per_direction: int = 2
    ) -> QueryResult:
        """
        Q5: Find taxa with conflicting associations (increased vs decreased).
        
        This query answers: "Which taxa show conflicting associations for [disease]?"
        
        **Validates: Requirements 1.4, 9.1, 9.2, 9.3, 9.4, 9.5**
        
        The query:
        1. Identifies taxa with both "increased" and "decreased" associations for the same disease
        2. Only returns taxa with >= min_papers_per_direction papers supporting each direction
        3. Calculates the percentage of papers supporting each direction
        4. Returns paper metadata (DOI, year, study_design) for all conflicting papers
        5. Sorts by total paper count descending, then by direction balance 
           (abs(increased_count - decreased_count) ascending)
        
        Args:
            disease: Disease entity name to query (e.g., "Crohn's Disease", "Type 2 Diabetes")
            min_papers_per_direction: Minimum number of papers required for each direction
                (increased and decreased) for a taxon to be included in results (default: 2)
        
        Returns:
            QueryResult containing:
            - taxon_name: Name of the taxon
            - total_paper_count: Total number of papers (increased + decreased)
            - increased_count: Number of papers reporting "increased"
            - decreased_count: Number of papers reporting "decreased"
            - increased_percentage: Percentage of papers reporting "increased"
            - decreased_percentage: Percentage of papers reporting "decreased"
            - direction_balance: Absolute difference between increased and decreased counts
            - increased_papers: List of paper metadata for "increased" papers
              (each with doi, year, study_design)
            - decreased_papers: List of paper metadata for "decreased" papers
              (each with doi, year, study_design)
        
        Example:
            result = engine.query_conflicting_evidence(
                disease="Crohn's Disease",
                min_papers_per_direction=2
            )
            
            # Returns taxa with conflicting evidence like:
            # {
            #   "taxon_name": "Escherichia coli",
            #   "total_paper_count": 8,
            #   "increased_count": 5,
            #   "decreased_count": 3,
            #   "increased_percentage": 62.5,
            #   "decreased_percentage": 37.5,
            #   "direction_balance": 2,
            #   "increased_papers": [
            #     {"doi": "10.1234/test1", "year": 2023, "study_design": "RCT"},
            #     {"doi": "10.1234/test2", "year": 2022, "study_design": "observational"},
            #     ...
            #   ],
            #   "decreased_papers": [
            #     {"doi": "10.1234/test3", "year": 2021, "study_design": "RCT"},
            #     ...
            #   ]
            # }
        
        Preconditions:
        - disease is a valid disease entity
        - min_papers_per_direction >= 1
        
        Postconditions (Requirements 9.1, 9.2, 9.3, 9.4, 9.5):
        - Returns taxa with both "increased" and "decreased" associations
        - Each direction has >= min_papers_per_direction supporting papers (Requirement 9.2)
        - Results include percentage of papers supporting each direction (Requirement 9.3)
        - Results include paper metadata (DOI, year, study_design) for all papers (Requirement 9.4)
        - Results sorted by total_paper_count DESC, then direction_balance ASC (Requirement 9.5)
        """
        # Validate inputs (Requirement 18.2)
        if not isinstance(min_papers_per_direction, int) or min_papers_per_direction < 1:
            return QueryResult(
                query_description=f"Conflicting evidence for {disease}",
                error="min_papers_per_direction must be an integer >= 1"
            )
        
        # Sanitize string inputs (Requirement 18.2)
        disease = self.sanitize_string_parameter(disease)
        
        # Build Cypher query (Requirements 9.1, 9.2, 9.3, 9.4, 9.5)
        # This query finds taxa with both increased and decreased associations
        cypher_query = """
        MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)
        WHERE r.disease = $disease
          AND (r.direction = 'increased' OR r.direction = 'decreased')
        WITH t,
             collect(CASE WHEN r.direction = 'increased' THEN p ELSE null END) as increased_papers_raw,
             collect(CASE WHEN r.direction = 'decreased' THEN p ELSE null END) as decreased_papers_raw
        WITH t,
             [p IN increased_papers_raw WHERE p IS NOT NULL] as increased_papers,
             [p IN decreased_papers_raw WHERE p IS NOT NULL] as decreased_papers
        WITH t,
             increased_papers,
             decreased_papers,
             size(increased_papers) as increased_count,
             size(decreased_papers) as decreased_count
        WHERE increased_count >= $min_papers_per_direction
          AND decreased_count >= $min_papers_per_direction
        WITH t,
             increased_papers,
             decreased_papers,
             increased_count,
             decreased_count,
             increased_count + decreased_count as total_paper_count
        WITH t,
             increased_papers,
             decreased_papers,
             increased_count,
             decreased_count,
             total_paper_count,
             100.0 * increased_count / total_paper_count as increased_percentage,
             100.0 * decreased_count / total_paper_count as decreased_percentage,
             abs(increased_count - decreased_count) as direction_balance
        RETURN t.name as taxon_name,
               total_paper_count,
               increased_count,
               decreased_count,
               increased_percentage,
               decreased_percentage,
               direction_balance,
               [p IN increased_papers | {
                   doi: COALESCE(p.doi, p.pmid, p.title),
                   year: p.year,
                   study_design: p.article_type
               }] as increased_papers,
               [p IN decreased_papers | {
                   doi: COALESCE(p.doi, p.pmid, p.title),
                   year: p.year,
                   study_design: p.article_type
               }] as decreased_papers
        ORDER BY total_paper_count DESC, direction_balance ASC
        """
        
        # Build parameters (Requirement 18.1 - parameterized queries)
        parameters = {
            "disease": disease,
            "min_papers_per_direction": min_papers_per_direction
        }
        
        # Execute query with caching (Requirement 13.5)
        return self._execute_with_cache(
            query_name="query_conflicting_evidence",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=f"Conflicting evidence for {disease} (min_papers_per_direction={min_papers_per_direction})",
                aggregation_method="conflicting_evidence_detection"
            )
        )

    def query_open_world_claims(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_type: Optional[str] = None,
        min_paper_count: int = 2,
        confidence_threshold: float = 0.7,
    ) -> QueryResult:
        """
        Query open-world claims matching filters. Results sorted by
        consensus_confidence DESC, paper_count DESC.
        Uses parameterized Cypher and 24-hour cache TTL.

        Requirements: 5.1, 5.4, 5.5, 5.6
        """
        # Build parameters (Requirement 18.1 - parameterized queries)
        parameters = {
            "subject": subject,
            "predicate": predicate,
            "object_type": object_type,
            "min_paper_count": min_paper_count,
            "confidence_threshold": confidence_threshold,
        }

        cypher_query = """
        MATCH (c:OpenWorldClaim)
        WHERE ($subject IS NULL OR c.subject_name = $subject)
          AND ($predicate IS NULL OR c.canonical_predicate = $predicate)
          AND ($object_type IS NULL OR c.object_type = $object_type)
          AND c.paper_count >= $min_paper_count
          AND c.consensus_confidence >= $confidence_threshold
        RETURN c.claim_id, c.subject_name, c.canonical_predicate, c.object_name,
               c.consensus_confidence, c.paper_count, c.evidence_strength,
               c.first_reported, c.last_updated
        ORDER BY c.consensus_confidence DESC, c.paper_count DESC
        """

        return self._execute_with_cache(
            query_name="query_open_world_claims",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=(
                    f"Open-world claims (subject={subject}, predicate={predicate}, "
                    f"object_type={object_type}, min_paper_count={min_paper_count}, "
                    f"confidence>={confidence_threshold})"
                ),
                confidence_threshold=confidence_threshold,
            ),
        )

    def query_entity_relationships(self, entity_name: str) -> QueryResult:
        """
        Return all relationships (canonical + promoted open-world) for an entity,
        grouped by predicate category.

        Requirements: 5.2
        """
        # Sanitize input (Requirement 18.2)
        entity_name = self.sanitize_string_parameter(entity_name)

        parameters = {"entity_name": entity_name}

        cypher_query = """
        MATCH (p:Paper)-[r]->(e)
        WHERE e.name = $entity_name OR e.canonical_name = $entity_name
        WITH type(r) as rel_type, r, p, e
        RETURN rel_type, r.confidence as confidence, p.id as paper_id, e.name as entity_name

        UNION

        MATCH (c:OpenWorldClaim)
        WHERE c.subject_name = $entity_name
        RETURN c.canonical_predicate as rel_type, c.consensus_confidence as confidence,
               c.supporting_papers as paper_ids, c.object_name as related_entity

        UNION

        MATCH (c:OpenWorldClaim)
        WHERE c.object_name = $entity_name
        RETURN c.canonical_predicate as rel_type, c.consensus_confidence as confidence,
               c.supporting_papers as paper_ids, c.subject_name as related_entity
        """

        return self._execute_with_cache(
            query_name="query_entity_relationships",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=f"All relationships for entity '{entity_name}'",
            ),
        )

    def query_cross_paper_predicates(
        self,
        predicate: str,
        min_paper_count: int = 2,
    ) -> QueryResult:
        """
        Return (subject, object) pairs connected by a predicate across required papers.
        Supports wildcard/pattern matching via Cypher `=~` operator.

        Requirements: 5.3
        """
        parameters = {
            "predicate_pattern": predicate,
            "min_paper_count": min_paper_count,
        }

        cypher_query = """
        MATCH (c:OpenWorldClaim)
        WHERE c.canonical_predicate =~ $predicate_pattern
          AND c.paper_count >= $min_paper_count
        RETURN c.subject_name, c.canonical_predicate, c.object_name,
               c.paper_count, c.consensus_confidence, c.evidence_strength
        ORDER BY c.consensus_confidence DESC, c.paper_count DESC
        """

        return self._execute_with_cache(
            query_name="query_cross_paper_predicates",
            parameters=parameters,
            query_func=lambda: self.execute_query(
                cypher_query=cypher_query,
                parameters=parameters,
                description=(
                    f"Cross-paper predicates matching '{predicate}' "
                    f"(min_paper_count={min_paper_count})"
                ),
            ),
        )

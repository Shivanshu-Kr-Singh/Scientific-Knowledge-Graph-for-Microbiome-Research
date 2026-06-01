"""
api/query_api.py
----------------
REST API endpoints for Scientific Knowledge Graph queries.

This module provides FastAPI endpoints that wrap the ResearchQueryEngine methods,
exposing all 5 research queries as HTTP POST endpoints with JSON request/response.

**Validates: Requirements 1.1, 1.2, 1.3**

Endpoints:
- POST /query/cross-study-associations: Find taxa with consistent disease associations
- POST /query/intervention-evidence: Find interventions with RCT-level evidence
- POST /query/methodology-landscape: Survey data availability and methodology trends
- POST /query/top-associations: Find top taxa by evidence quality
- POST /query/conflicting-evidence: Find taxa with conflicting associations

Security Features (Requirement 18):
- Input validation using Pydantic models (Requirement 18.2)
- Parameterized queries (handled by ResearchQueryEngine) (Requirement 18.1)
- Rate limiting: 10 queries per minute per user (Requirement 18.4)
- Query complexity limits: max 1000 results, max depth 5 (Requirement 18.3)
- Request/response logging

Usage:
    # Start the API server
    uvicorn api.query_api:app --host 0.0.0.0 --port 8000
    
    # Example request
    curl -X POST http://localhost:8000/query/cross-study-associations \
         -H "Content-Type: application/json" \
         -d '{"disease": "Type 2 Diabetes", "study_type": "RCT", "min_papers": 3}'
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from graph.research_query_engine import ResearchQueryEngine, QueryResult
from api.input_validator import InputValidator, create_error_response
from api.rate_limiter import rate_limiter
from api.query_complexity_limiter import query_complexity_limiter
from loguru import logger

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Scientific Knowledge Graph API",
    description="REST API for querying the microbiome research knowledge graph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize Neo4j driver and query engine
# These will be initialized on startup
neo4j_driver = None
query_engine = None
input_validator = None


# ============================================================================
# Rate Limiting Dependency
# ============================================================================

async def check_rate_limit(request: Request):
    """
    Dependency to check rate limits before processing requests.
    
    **Validates: Requirement 18.4**
    
    Raises:
        HTTPException: 429 Too Many Requests if rate limit exceeded
    """
    await rate_limiter(request)


@app.on_event("startup")
async def startup_event():
    """
    Initialize Neo4j connection and query engine on startup.
    
    **Validates: Requirement 1.1, 1.2, 1.3, 18.2**
    """
    global neo4j_driver, query_engine, input_validator
    
    # Get Neo4j connection details from environment
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        # Initialize Neo4j driver
        neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        # Verify connection
        neo4j_driver.verify_connectivity()
        
        # Initialize query engine with caching enabled
        query_engine = ResearchQueryEngine(
            driver=neo4j_driver,
            enable_cache=True,
            cache_ttl_hours=24
        )
        
        # Initialize input validator with Neo4j driver for entity validation
        input_validator = InputValidator(neo4j_driver=neo4j_driver)
        
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        logger.info("Query engine initialized with caching enabled")
        logger.info("Input validator initialized with entity validation")
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Close Neo4j connection on shutdown.
    """
    global neo4j_driver
    
    if neo4j_driver:
        neo4j_driver.close()
        logger.info("Neo4j connection closed")


# ============================================================================
# Request/Response Models
# ============================================================================

class CrossStudyAssociationsRequest(BaseModel):
    """
    Request model for cross-study associations query.
    
    **Validates: Requirement 6.1**
    """
    disease: str = Field(
        ...,
        description="Disease entity name (e.g., 'Type 2 Diabetes', 'IBD')",
        min_length=1,
        max_length=200
    )
    study_type: str = Field(
        default="RCT",
        description="Type of study: 'RCT', 'observational', 'meta_analysis', or 'any'"
    )
    min_papers: int = Field(
        default=3,
        ge=1,
        description="Minimum number of papers required for a taxon"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0.0-1.0)"
    )
    require_open_data: bool = Field(
        default=True,
        description="Only include papers with open data"
    )
    
    @field_validator("study_type")
    @classmethod
    def validate_study_type(cls, v):
        allowed = ["RCT", "observational", "meta_analysis", "any"]
        if v not in allowed:
            raise ValueError(f"study_type must be one of {allowed}")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "disease": "Type 2 Diabetes",
                "study_type": "RCT",
                "min_papers": 3,
                "confidence_threshold": 0.7,
                "require_open_data": True
            }
        }
    )


class InterventionEvidenceRequest(BaseModel):
    """
    Request model for intervention evidence query.
    
    **Validates: Requirement 7.1**
    """
    intervention_types: List[str] = Field(
        ...,
        description="List of intervention types (e.g., ['probiotic', 'FMT', 'diet'])",
        min_length=1
    )
    min_sample_size: int = Field(
        default=50,
        ge=1,
        description="Minimum total sample size"
    )
    evidence_strength: str = Field(
        default="strong",
        description="Minimum evidence strength: 'strong', 'moderate', 'weak', or 'any'"
    )
    
    @field_validator("evidence_strength")
    @classmethod
    def validate_evidence_strength(cls, v):
        allowed = ["strong", "moderate", "weak", "any"]
        if v not in allowed:
            raise ValueError(f"evidence_strength must be one of {allowed}")
        return v
    
    @field_validator("intervention_types")
    @classmethod
    def validate_intervention_types(cls, v):
        if not v or len(v) == 0:
            raise ValueError("intervention_types must be a non-empty list")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intervention_types": ["probiotic", "FMT"],
                "min_sample_size": 50,
                "evidence_strength": "strong"
            }
        }
    )


class MethodologyLandscapeRequest(BaseModel):
    """
    Request model for methodology landscape query.
    
    **Validates: Requirement 8.1**
    """
    year_start: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Start year of the time period (inclusive)"
    )
    year_end: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="End year of the time period (inclusive)"
    )
    sequencing_methods: List[str] = Field(
        ...,
        description="List of sequencing methods (e.g., ['16S rRNA sequencing', 'shotgun metagenomics'])",
        min_length=1
    )
    require_deposited_data: bool = Field(
        default=True,
        description="Only include papers with deposited data"
    )
    
    @field_validator("sequencing_methods")
    @classmethod
    def validate_sequencing_methods(cls, v):
        if not v or len(v) == 0:
            raise ValueError("sequencing_methods must be a non-empty list")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "year_start": 2020,
                "year_end": 2024,
                "sequencing_methods": ["16S rRNA sequencing", "shotgun metagenomics"],
                "require_deposited_data": True
            }
        }
    )


class TopAssociationsRequest(BaseModel):
    """
    Request model for top associations query.
    
    **Validates: Requirement 1.4, 18.3**
    """
    disease: str = Field(
        ...,
        description="Disease entity name (e.g., 'Type 2 Diabetes', 'IBD')",
        min_length=1,
        max_length=200
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=1000,  # Max result count limit (Requirement 18.3)
        description="Maximum number of taxa to return (max 1000)"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0.0-1.0)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "disease": "IBD",
                "top_n": 10,
                "min_confidence": 0.7
            }
        }
    )


class ConflictingEvidenceRequest(BaseModel):
    """
    Request model for conflicting evidence query.
    
    **Validates: Requirement 9.1**
    """
    disease: str = Field(
        ...,
        description="Disease entity name (e.g., 'Crohn\\'s Disease', 'Type 2 Diabetes')",
        min_length=1,
        max_length=200
    )
    min_papers_per_direction: int = Field(
        default=2,
        ge=1,
        description="Minimum number of papers required for each direction"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "disease": "Crohn's Disease",
                "min_papers_per_direction": 2
            }
        }
    )


class QueryResponse(BaseModel):
    """
    Response model for all query endpoints.
    
    This wraps the QueryResult from ResearchQueryEngine with additional
    API-level metadata.
    
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    success: bool = Field(
        ...,
        description="Whether the query executed successfully"
    )
    query_result: Optional[QueryResult] = Field(
        default=None,
        description="Query result with data and metadata"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if query failed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "query_result": {
                    "query_id": "550e8400-e29b-41d4-a716-446655440000",
                    "query_description": "Cross-study associations for Type 2 Diabetes",
                    "results": [
                        {
                            "taxon_name": "Bacteroides fragilis",
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
                },
                "error": None
            }
        }
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Scientific Knowledge Graph API",
        "version": "1.0.0",
        "description": "REST API for querying the microbiome research knowledge graph",
        "endpoints": {
            "cross_study_associations": "/query/cross-study-associations",
            "intervention_evidence": "/query/intervention-evidence",
            "methodology_landscape": "/query/methodology-landscape",
            "top_associations": "/query/top-associations",
            "conflicting_evidence": "/query/conflicting-evidence"
        },
        "management": {
            "health": "/health",
            "cache_stats": "/cache/stats",
            "cache_invalidate": "/cache/invalidate",
            "limits": "/limits"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "security": {
            "rate_limiting": "10 queries per minute per user",
            "max_results": "1000 results per query",
            "max_query_depth": 5
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and Neo4j connectivity.
    """
    global neo4j_driver, query_engine
    
    if not neo4j_driver or not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    try:
        # Verify Neo4j connection
        neo4j_driver.verify_connectivity()
        
        # Get cache stats if available
        cache_stats = query_engine.get_cache_stats()
        
        return {
            "status": "healthy",
            "neo4j": "connected",
            "cache": cache_stats if cache_stats else "disabled"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Neo4j connection failed: {str(e)}"
        )


@app.post("/query/cross-study-associations", response_model=QueryResponse)
async def query_cross_study_associations(
    request: CrossStudyAssociationsRequest,
    _: None = Depends(check_rate_limit)
):
    """
    Q1: Find taxa with consistent disease associations across multiple studies.
    
    This endpoint answers: "Which gut microbiome taxa show consistent association 
    with [disease] across [study_type] studies with open sequencing data?"
    
    **Validates: Requirements 1.1, 6.1, 6.2, 6.3, 6.4, 6.5, 18.2, 18.3, 18.4**
    
    Args:
        request: CrossStudyAssociationsRequest with query parameters
    
    Returns:
        QueryResponse with taxa and their consensus metrics
    
    Example:
        POST /query/cross-study-associations
        {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": true
        }
    """
    global query_engine, input_validator
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    if not input_validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Input validator not initialized"
        )
    
    # Validate inputs
    errors = []
    
    # Validate disease name
    error = input_validator.validate_entity_name(
        request.disease,
        "disease",
        "disease",
        check_existence=True
    )
    if error:
        errors.append(error)
    
    # Validate numeric thresholds
    error = input_validator.validate_numeric_threshold(
        request.min_papers,
        "min_papers",
        input_validator.MIN_PAPERS,
        input_validator.MAX_PAPERS
    )
    if error:
        errors.append(error)
    
    error = input_validator.validate_numeric_threshold(
        request.confidence_threshold,
        "confidence_threshold",
        input_validator.MIN_CONFIDENCE,
        input_validator.MAX_CONFIDENCE
    )
    if error:
        errors.append(error)
    
    # If validation failed, return 400 Bad Request with details
    if errors:
        logger.warning(f"Validation failed for cross-study associations query: {len(errors)} errors")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_error_response(errors)
        )
    
    try:
        logger.info(f"Executing cross-study associations query for disease: {request.disease}")
        
        # Execute query using ResearchQueryEngine
        result = query_engine.query_cross_study_associations(
            disease=request.disease,
            study_type=request.study_type,
            min_papers=request.min_papers,
            confidence_threshold=request.confidence_threshold,
            require_open_data=request.require_open_data
        )
        
        # Check for errors
        if result.error:
            logger.error(f"Query failed: {result.error}")
            return QueryResponse(
                success=False,
                query_result=result,
                error=result.error
            )
        
        logger.info(f"Query completed successfully: {result.result_count} results in {result.execution_time_ms:.2f}ms")
        
        return QueryResponse(
            success=True,
            query_result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in cross-study associations query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/query/intervention-evidence", response_model=QueryResponse)
async def query_intervention_evidence(
    request: InterventionEvidenceRequest,
    _: None = Depends(check_rate_limit)
):
    """
    Q2: Find interventions with RCT-level evidence for modifying specific taxa.
    
    This endpoint answers: "What interventions (probiotics, FMT, diet) have 
    RCT-level evidence for modifying specific gut taxa, and what effect 
    directions are reported?"
    
    **Validates: Requirements 1.2, 7.1, 7.2, 7.3, 7.4, 7.5, 18.2, 18.3, 18.4**
    
    Args:
        request: InterventionEvidenceRequest with query parameters
    
    Returns:
        QueryResponse with interventions and their evidence metrics
    
    Example:
        POST /query/intervention-evidence
        {
            "intervention_types": ["probiotic", "FMT"],
            "min_sample_size": 50,
            "evidence_strength": "strong"
        }
    """
    global query_engine, input_validator
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    if not input_validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Input validator not initialized"
        )
    
    # Validate inputs
    errors = []
    
    # Validate intervention types list
    list_errors = input_validator.validate_string_list(
        request.intervention_types,
        "intervention_types",
        entity_type=None,  # Don't validate as entities, just sanitize
        check_existence=False
    )
    errors.extend(list_errors)
    
    # Validate min_sample_size
    error = input_validator.validate_numeric_threshold(
        request.min_sample_size,
        "min_sample_size",
        input_validator.MIN_SAMPLE_SIZE,
        input_validator.MAX_SAMPLE_SIZE
    )
    if error:
        errors.append(error)
    
    # If validation failed, return 400 Bad Request with details
    if errors:
        logger.warning(f"Validation failed for intervention evidence query: {len(errors)} errors")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_error_response(errors)
        )
    
    try:
        logger.info(f"Executing intervention evidence query for types: {request.intervention_types}")
        
        # Execute query using ResearchQueryEngine
        result = query_engine.query_intervention_evidence(
            intervention_types=request.intervention_types,
            min_sample_size=request.min_sample_size,
            evidence_strength=request.evidence_strength
        )
        
        # Check for errors
        if result.error:
            logger.error(f"Query failed: {result.error}")
            return QueryResponse(
                success=False,
                query_result=result,
                error=result.error
            )
        
        logger.info(f"Query completed successfully: {result.result_count} results in {result.execution_time_ms:.2f}ms")
        
        return QueryResponse(
            success=True,
            query_result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in intervention evidence query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/query/methodology-landscape", response_model=QueryResponse)
async def query_methodology_landscape(
    request: MethodologyLandscapeRequest,
    _: None = Depends(check_rate_limit)
):
    """
    Q3: Survey data availability and methodology across time period.
    
    This endpoint answers: "Which microbiome studies from [year_start]-[year_end] 
    deposited data on SRA/ENA and used shotgun metagenomics vs 16S sequencing?"
    
    **Validates: Requirements 1.3, 8.1, 8.2, 8.3, 8.4, 8.5, 18.2, 18.3, 18.4**
    
    Args:
        request: MethodologyLandscapeRequest with query parameters
    
    Returns:
        QueryResponse with methodology trends and data availability metrics
    
    Example:
        POST /query/methodology-landscape
        {
            "year_start": 2020,
            "year_end": 2024,
            "sequencing_methods": ["16S rRNA sequencing", "shotgun metagenomics"],
            "require_deposited_data": true
        }
    """
    global query_engine, input_validator
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    if not input_validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Input validator not initialized"
        )
    
    # Validate inputs
    errors = []
    
    # Validate year range
    year_errors = input_validator.validate_year_range(
        request.year_start,
        request.year_end
    )
    errors.extend(year_errors)
    
    # Validate sequencing methods list
    list_errors = input_validator.validate_string_list(
        request.sequencing_methods,
        "sequencing_methods",
        entity_type="method",
        check_existence=True
    )
    errors.extend(list_errors)
    
    # If validation failed, return 400 Bad Request with details
    if errors:
        logger.warning(f"Validation failed for methodology landscape query: {len(errors)} errors")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_error_response(errors)
        )
    
    try:
        logger.info(f"Executing methodology landscape query for years {request.year_start}-{request.year_end}")
        
        # Execute query using ResearchQueryEngine
        result = query_engine.query_methodology_landscape(
            year_start=request.year_start,
            year_end=request.year_end,
            sequencing_methods=request.sequencing_methods,
            require_deposited_data=request.require_deposited_data
        )
        
        # Check for errors
        if result.error:
            logger.error(f"Query failed: {result.error}")
            return QueryResponse(
                success=False,
                query_result=result,
                error=result.error
            )
        
        logger.info(f"Query completed successfully: {result.result_count} results in {result.execution_time_ms:.2f}ms")
        
        return QueryResponse(
            success=True,
            query_result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in methodology landscape query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/query/top-associations", response_model=QueryResponse)
async def query_top_associations(
    request: TopAssociationsRequest,
    _: None = Depends(check_rate_limit)
):
    """
    Q4: Find top taxa associated with a disease ranked by evidence quality.
    
    This endpoint answers: "Top N taxa associated with [disease] across multiple 
    papers with confidence >= [min_confidence], ranked by evidence quality."
    
    **Validates: Requirements 1.4, 18.2, 18.3, 18.4**
    
    Args:
        request: TopAssociationsRequest with query parameters
    
    Returns:
        QueryResponse with top taxa ranked by evidence quality
    
    Example:
        POST /query/top-associations
        {
            "disease": "IBD",
            "top_n": 10,
            "min_confidence": 0.7
        }
    """
    global query_engine, input_validator
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    if not input_validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Input validator not initialized"
        )
    
    # Validate inputs
    errors = []
    
    # Validate disease name
    error = input_validator.validate_entity_name(
        request.disease,
        "disease",
        "disease",
        check_existence=True
    )
    if error:
        errors.append(error)
    
    # Validate top_n
    error = input_validator.validate_numeric_threshold(
        request.top_n,
        "top_n",
        input_validator.MIN_TOP_N,
        input_validator.MAX_TOP_N
    )
    if error:
        errors.append(error)
    
    # Validate min_confidence
    error = input_validator.validate_numeric_threshold(
        request.min_confidence,
        "min_confidence",
        input_validator.MIN_CONFIDENCE,
        input_validator.MAX_CONFIDENCE
    )
    if error:
        errors.append(error)
    
    # If validation failed, return 400 Bad Request with details
    if errors:
        logger.warning(f"Validation failed for top associations query: {len(errors)} errors")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_error_response(errors)
        )
    
    try:
        logger.info(f"Executing top associations query for disease: {request.disease}")
        
        # Apply query complexity limit to top_n parameter (Requirement 18.3)
        limited_top_n = query_complexity_limiter.validate_result_count_limit(request.top_n)
        
        # Execute query using ResearchQueryEngine
        result = query_engine.query_top_associations_by_evidence(
            disease=request.disease,
            top_n=limited_top_n,
            min_confidence=request.min_confidence
        )
        
        # Check for errors
        if result.error:
            logger.error(f"Query failed: {result.error}")
            return QueryResponse(
                success=False,
                query_result=result,
                error=result.error
            )
        
        logger.info(f"Query completed successfully: {result.result_count} results in {result.execution_time_ms:.2f}ms")
        
        return QueryResponse(
            success=True,
            query_result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in top associations query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/query/conflicting-evidence", response_model=QueryResponse)
async def query_conflicting_evidence(
    request: ConflictingEvidenceRequest,
    _: None = Depends(check_rate_limit)
):
    """
    Q5: Find taxa with conflicting associations (increased vs decreased).
    
    This endpoint answers: "Which taxa show conflicting associations for [disease]?"
    
    **Validates: Requirements 1.4, 9.1, 9.2, 9.3, 9.4, 9.5, 18.2, 18.3, 18.4**
    
    Args:
        request: ConflictingEvidenceRequest with query parameters
    
    Returns:
        QueryResponse with taxa showing conflicting evidence
    
    Example:
        POST /query/conflicting-evidence
        {
            "disease": "Crohn's Disease",
            "min_papers_per_direction": 2
        }
    """
    global query_engine, input_validator
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    if not input_validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Input validator not initialized"
        )
    
    # Validate inputs
    errors = []
    
    # Validate disease name
    error = input_validator.validate_entity_name(
        request.disease,
        "disease",
        "disease",
        check_existence=True
    )
    if error:
        errors.append(error)
    
    # Validate min_papers_per_direction
    error = input_validator.validate_numeric_threshold(
        request.min_papers_per_direction,
        "min_papers_per_direction",
        input_validator.MIN_PAPERS,
        input_validator.MAX_PAPERS
    )
    if error:
        errors.append(error)
    
    # If validation failed, return 400 Bad Request with details
    if errors:
        logger.warning(f"Validation failed for conflicting evidence query: {len(errors)} errors")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_error_response(errors)
        )
    
    try:
        logger.info(f"Executing conflicting evidence query for disease: {request.disease}")
        
        # Execute query using ResearchQueryEngine
        result = query_engine.query_conflicting_evidence(
            disease=request.disease,
            min_papers_per_direction=request.min_papers_per_direction
        )
        
        # Check for errors
        if result.error:
            logger.error(f"Query failed: {result.error}")
            return QueryResponse(
                success=False,
                query_result=result,
                error=result.error
            )
        
        logger.info(f"Query completed successfully: {result.result_count} results in {result.execution_time_ms:.2f}ms")
        
        return QueryResponse(
            success=True,
            query_result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in conflicting evidence query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/cache/invalidate")
async def invalidate_cache():
    """
    Invalidate all cached query results and validation cache.
    
    This endpoint should be called when new data is loaded into the knowledge graph
    to ensure queries return fresh results and entity validation is up-to-date.
    
    **Validates: Requirement 13.5, 18.2**
    
    Returns:
        Number of cache entries that were invalidated
    
    Example:
        POST /cache/invalidate
    """
    global query_engine, input_validator
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    try:
        # Invalidate query cache
        invalidated_count = query_engine.invalidate_cache()
        logger.info(f"Query cache invalidated: {invalidated_count} entries removed")
        
        # Invalidate validation cache
        if input_validator:
            input_validator.invalidate_cache()
            logger.info("Validation cache invalidated")
        
        return {
            "success": True,
            "invalidated_count": invalidated_count,
            "message": f"Successfully invalidated {invalidated_count} query cache entries and validation cache"
        }
        
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache invalidation failed: {str(e)}"
        )


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns cache hit rate, size, and other metrics.
    
    **Validates: Requirement 13.5**
    
    Returns:
        Cache statistics dictionary
    
    Example:
        GET /cache/stats
    """
    global query_engine
    
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    try:
        stats = query_engine.get_cache_stats()
        
        if stats is None:
            return {
                "success": True,
                "cache_enabled": False,
                "message": "Caching is disabled"
            }
        
        return {
            "success": True,
            "cache_enabled": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@app.get("/limits")
async def get_limits():
    """
    Get API rate limits and query complexity limits.
    
    Returns current limits for rate limiting and query complexity.
    
    **Validates: Requirements 18.3, 18.4**
    
    Returns:
        Dictionary with rate limits and query complexity limits
    
    Example:
        GET /limits
    """
    try:
        rate_limit_stats = rate_limiter.get_stats()
        complexity_limits = query_complexity_limiter.get_limits()
        
        return {
            "success": True,
            "rate_limiting": {
                "max_requests_per_window": rate_limit_stats["max_requests_per_window"],
                "window_seconds": rate_limit_stats["window_seconds"],
                "description": f"{rate_limit_stats['max_requests_per_window']} queries per {rate_limit_stats['window_seconds']} seconds per user"
            },
            "query_complexity": {
                "max_result_count": complexity_limits["max_result_count"],
                "max_query_depth": complexity_limits["max_query_depth"],
                "description": f"Maximum {complexity_limits['max_result_count']} results per query, maximum depth {complexity_limits['max_query_depth']}"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get limits: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api.query_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

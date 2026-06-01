"""
graph/extractor_registry.py
----------------------------
Registry of extraction methods for knowledge graph relationship extraction.

This module maintains a registry of all extraction methods with unique identifiers,
versions, and source code hashes for reproducibility and rollback capabilities.

Requirements: 10.1, 10.2, 10.3, 10.4, 19.1
"""

import hashlib
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ExtractorMetadata(BaseModel):
    """
    Metadata for a registered extraction method.
    
    Stores version information, source code hash, and registration details
    for reproducibility and auditability.
    
    Requirements: 10.1, 10.3, 10.4, 19.1
    """
    
    method_id: str = Field(..., description="Unique identifier (e.g., 'regex_ner', 'llm_extractor_v1.2')")
    method_type: str = Field(..., description="Type of extractor: regex | ml_model | llm | manual")
    version: str = Field(..., description="Version string (e.g., '1.0', '1.2', '2.0')")
    description: str = Field(..., description="Human-readable description of the method")
    
    # Reproducibility (Requirements 10.4, 19.1)
    source_code_hash: Optional[str] = Field(None, description="SHA-256 hash of source code for reproducibility")
    model_hash: Optional[str] = Field(None, description="Hash of ML model weights if applicable")
    prompt_template_hash: Optional[str] = Field(None, description="Hash of LLM prompt template if applicable")
    
    # Registration metadata
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    registered_by: str = Field(default="system", description="User ID who registered this method")
    is_active: bool = Field(default=True, description="Whether this method is currently active")
    
    # Performance metadata
    avg_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average confidence score")
    total_extractions: int = Field(default=0, description="Total number of relationships extracted")


class ExtractorRegistry:
    """
    Registry of extraction methods for knowledge graph construction.
    
    Maintains a registry of all extraction methods with unique identifiers,
    versions, and source code hashes. Provides validation and querying
    capabilities for reproducibility and rollback.
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 19.1
    """
    
    def __init__(self):
        """Initialize the extractor registry with default methods."""
        self._registry: Dict[str, ExtractorMetadata] = {}
        self._initialize_default_extractors()
    
    def _initialize_default_extractors(self):
        """
        Initialize registry with default extraction methods.
        
        Requirement 10.1: System SHALL maintain a registry of all extraction
        methods with unique identifiers.
        """
        default_extractors = [
            ExtractorMetadata(
                method_id="regex_ner",
                method_type="regex",
                version="1.0",
                description="Regular expression-based named entity recognition for taxa and diseases",
                source_code_hash=None,  # Will be computed from actual source
                registered_by="system",
            ),
            ExtractorMetadata(
                method_id="biobert_ner",
                method_type="ml_model",
                version="1.0",
                description="BioBERT-based named entity recognition for biomedical entities",
                source_code_hash=None,
                model_hash=None,  # Will be computed from model weights
                registered_by="system",
            ),
            ExtractorMetadata(
                method_id="llm_extractor_v1.0",
                method_type="llm",
                version="1.0",
                description="LLM-based relationship extraction (initial version)",
                source_code_hash=None,
                prompt_template_hash=None,  # Will be computed from prompt template
                registered_by="system",
            ),
            ExtractorMetadata(
                method_id="llm_extractor_v1.1",
                method_type="llm",
                version="1.1",
                description="LLM-based relationship extraction with improved prompts",
                source_code_hash=None,
                prompt_template_hash=None,
                registered_by="system",
            ),
            ExtractorMetadata(
                method_id="llm_extractor_v1.2",
                method_type="llm",
                version="1.2",
                description="LLM-based relationship extraction with statistical measure parsing",
                source_code_hash=None,
                prompt_template_hash=None,
                registered_by="system",
            ),
            ExtractorMetadata(
                method_id="manual_curation",
                method_type="manual",
                version="1.0",
                description="Manually curated relationships by domain experts",
                registered_by="system",
            ),
            ExtractorMetadata(
                method_id="legacy",
                method_type="manual",
                version="1.0",
                description="Legacy relationships from old system without provenance",
                registered_by="system",
            ),
        ]
        
        for extractor in default_extractors:
            self._registry[extractor.method_id] = extractor
    
    def register_extractor(
        self,
        method_id: str,
        method_type: str,
        version: str,
        description: str,
        source_code: Optional[str] = None,
        model_weights: Optional[bytes] = None,
        prompt_template: Optional[str] = None,
        registered_by: str = "system",
    ) -> ExtractorMetadata:
        """
        Register a new extraction method in the registry.
        
        Requirement 10.1: System SHALL maintain a registry of all extraction
        methods with unique identifiers.
        
        Requirement 19.1: System SHALL store extraction method source code hash
        for reproducibility.
        
        Args:
            method_id: Unique identifier for the method
            method_type: Type of extractor (regex, ml_model, llm, manual)
            version: Version string
            description: Human-readable description
            source_code: Source code string to hash (optional)
            model_weights: Model weights bytes to hash (optional)
            prompt_template: LLM prompt template to hash (optional)
            registered_by: User ID registering this method
        
        Returns:
            ExtractorMetadata for the registered method
        
        Raises:
            ValueError: If method_id already exists or method_type is invalid
        """
        if method_id in self._registry:
            raise ValueError(f"Extractor '{method_id}' is already registered")
        
        valid_types = {"regex", "ml_model", "llm", "manual"}
        if method_type not in valid_types:
            raise ValueError(f"method_type must be one of {valid_types}, got '{method_type}'")
        
        # Compute hashes for reproducibility (Requirement 19.1)
        source_code_hash = self._compute_hash(source_code) if source_code else None
        model_hash = self._compute_hash(model_weights) if model_weights else None
        prompt_template_hash = self._compute_hash(prompt_template) if prompt_template else None
        
        metadata = ExtractorMetadata(
            method_id=method_id,
            method_type=method_type,
            version=version,
            description=description,
            source_code_hash=source_code_hash,
            model_hash=model_hash,
            prompt_template_hash=prompt_template_hash,
            registered_by=registered_by,
        )
        
        self._registry[method_id] = metadata
        return metadata
    
    def is_registered(self, method_id: str) -> bool:
        """
        Check if an extraction method is registered.
        
        Requirement 10.2: System SHALL validate that extraction_method exists
        in the registered extractors list before allowing relationship creation.
        
        Args:
            method_id: The method identifier to check
        
        Returns:
            True if method is registered, False otherwise
        """
        return method_id in self._registry
    
    def get_extractor(self, method_id: str) -> Optional[ExtractorMetadata]:
        """
        Get metadata for a registered extraction method.
        
        Requirement 10.3: System SHALL record extractor_version for every relationship.
        
        Args:
            method_id: The method identifier
        
        Returns:
            ExtractorMetadata if found, None otherwise
        """
        return self._registry.get(method_id)
    
    def get_all_extractors(self) -> List[ExtractorMetadata]:
        """
        Get all registered extraction methods.
        
        Returns:
            List of all ExtractorMetadata objects
        """
        return list(self._registry.values())
    
    def get_active_extractors(self) -> List[ExtractorMetadata]:
        """
        Get all active extraction methods.
        
        Returns:
            List of active ExtractorMetadata objects
        """
        return [meta for meta in self._registry.values() if meta.is_active]
    
    def get_method_ids(self) -> Set[str]:
        """
        Get set of all registered method identifiers.
        
        Requirement 10.1: System SHALL maintain a registry of all extraction
        methods with unique identifiers.
        
        Returns:
            Set of method IDs
        """
        return set(self._registry.keys())
    
    def deactivate_extractor(self, method_id: str) -> bool:
        """
        Deactivate an extraction method (for rollback scenarios).
        
        Requirement 10.5: System SHALL support rollback of extractions by method version.
        
        Args:
            method_id: The method identifier to deactivate
        
        Returns:
            True if deactivated, False if not found
        """
        if method_id in self._registry:
            self._registry[method_id].is_active = False
            return True
        return False
    
    def update_extractor_stats(
        self,
        method_id: str,
        avg_confidence: Optional[float] = None,
        total_extractions: Optional[int] = None,
    ) -> bool:
        """
        Update performance statistics for an extraction method.
        
        Args:
            method_id: The method identifier
            avg_confidence: Average confidence score
            total_extractions: Total number of extractions
        
        Returns:
            True if updated, False if not found
        """
        if method_id not in self._registry:
            return False
        
        metadata = self._registry[method_id]
        if avg_confidence is not None:
            metadata.avg_confidence = avg_confidence
        if total_extractions is not None:
            metadata.total_extractions = total_extractions
        
        return True
    
    def validate_extraction_method(self, method_id: str) -> bool:
        """
        Validate that an extraction method exists and is active.
        
        Requirement 10.2: System SHALL validate that extraction_method exists
        in the registered extractors list before allowing relationship creation.
        
        Args:
            method_id: The method identifier to validate
        
        Returns:
            True if method exists and is active, False otherwise
        """
        if method_id not in self._registry:
            return False
        return self._registry[method_id].is_active
    
    def get_extractors_by_type(self, method_type: str) -> List[ExtractorMetadata]:
        """
        Get all extractors of a specific type.
        
        Args:
            method_type: Type of extractor (regex, ml_model, llm, manual)
        
        Returns:
            List of ExtractorMetadata objects matching the type
        """
        return [
            meta for meta in self._registry.values()
            if meta.method_type == method_type
        ]
    
    def compute_source_hash(self, source_code: str) -> str:
        """
        Compute SHA-256 hash of source code for reproducibility.
        
        Requirement 19.1: System SHALL store extraction method source code hash
        for reproducibility.
        
        Args:
            source_code: Source code string to hash
        
        Returns:
            Hexadecimal SHA-256 hash string
        """
        return self._compute_hash(source_code)
    
    def compute_prompt_hash(self, prompt_template: str) -> str:
        """
        Compute SHA-256 hash of LLM prompt template.
        
        Requirement 19.2: System SHALL store LLM prompt hash for all
        LLM-based extractions.
        
        Args:
            prompt_template: LLM prompt template string to hash
        
        Returns:
            Hexadecimal SHA-256 hash string
        """
        return self._compute_hash(prompt_template)
    
    @staticmethod
    def _compute_hash(data: str | bytes) -> str:
        """
        Compute SHA-256 hash of data.
        
        Args:
            data: String or bytes to hash
        
        Returns:
            Hexadecimal SHA-256 hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()


# Global registry instance
# Requirement 10.1: System SHALL maintain a registry of all extraction methods
_global_registry = ExtractorRegistry()


def get_registry() -> ExtractorRegistry:
    """
    Get the global extractor registry instance.
    
    Returns:
        The global ExtractorRegistry instance
    """
    return _global_registry


def is_valid_extraction_method(method_id: str) -> bool:
    """
    Check if an extraction method is valid and registered.
    
    Requirement 10.2: System SHALL validate that extraction_method exists
    in the registered extractors list before allowing relationship creation.
    
    Args:
        method_id: The method identifier to validate
    
    Returns:
        True if method is registered and active, False otherwise
    """
    return _global_registry.validate_extraction_method(method_id)


def get_registered_method_ids() -> Set[str]:
    """
    Get set of all registered extraction method identifiers.
    
    Requirement 10.1: System SHALL maintain a registry of all extraction
    methods with unique identifiers.
    
    Returns:
        Set of registered method IDs
    """
    return _global_registry.get_method_ids()

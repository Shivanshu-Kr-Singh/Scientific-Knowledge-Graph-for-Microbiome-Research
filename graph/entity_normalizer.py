import yaml
import sqlite3
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from Bio import Entrez

SCHEMA = Path(__file__).parent / "schemas"
DB_PATH = Path(__file__).parent / "entity_normalization.db"

# Configure Entrez email for NCBI API
Entrez.email = "your_email@example.com"  # Should be configured via environment variable

class EntityNormalizer:
    def __init__(self):
        self.taxa = self._load("taxonomy_map.yaml")
        self.disease = self._load("disease_map.yaml")
        self._init_failure_log_db()

    def _load(self, fname):
        path = SCHEMA / fname

        if not path.exists():
            return {}

        with open(path) as fp:
            return yaml.safe_load(fp) or {}

    def _init_failure_log_db(self):
        """Initialize SQLite database for logging normalization failures"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_normalization_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                failure_reason TEXT NOT NULL,
                attempted_matches TEXT,
                timestamp TEXT NOT NULL,
                grounded BOOLEAN DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def _log_failure(self, entity_text: str, entity_type: str, failure_reason: str, attempted_matches: str = ""):
        """Log normalization failure to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO entity_normalization_failures 
            (entity_text, entity_type, failure_reason, attempted_matches, timestamp, grounded)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (entity_text, entity_type, failure_reason, attempted_matches, datetime.now(timezone.utc).isoformat(), False))
        conn.commit()
        conn.close()

    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _fuzzy_match_local(self, entity_text: str, mapping: Dict[str, list], max_distance: int = 2) -> Optional[str]:
        """Attempt fuzzy matching against local mapping with edit distance <= max_distance"""
        entity_lower = entity_text.lower()
        
        for canonical, variants in mapping.items():
            for variant in variants:
                if self._calculate_edit_distance(entity_lower, variant.lower()) <= max_distance:
                    return canonical
        
        return None

    def _ground_taxon_ncbi(self, taxon_name: str) -> Optional[Dict[str, Any]]:
        """Ground taxon to NCBI Taxonomy using Entrez API"""
        try:
            # Search NCBI Taxonomy database
            handle = Entrez.esearch(db="taxonomy", term=taxon_name, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            
            if record["IdList"]:
                taxon_id = record["IdList"][0]
                
                # Fetch detailed information
                handle = Entrez.efetch(db="taxonomy", id=taxon_id, retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                
                if records:
                    taxon_record = records[0]
                    return {
                        "id": f"ncbi:{taxon_id}",
                        "canonical_name": taxon_record.get("ScientificName", taxon_name),
                        "ontology": "NCBI Taxonomy",
                        "rank": taxon_record.get("Rank", "unknown"),
                        "grounded": True
                    }
            
            return None
        except Exception as e:
            # Log error but don't crash
            print(f"Error grounding taxon to NCBI: {e}")
            return None

    def _ground_disease_mesh(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """Ground disease to MeSH ontology using NCBI Entrez API"""
        try:
            # Search MeSH database
            handle = Entrez.esearch(db="mesh", term=disease_name, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            
            if record["IdList"]:
                mesh_id = record["IdList"][0]
                
                # Fetch detailed information
                handle = Entrez.efetch(db="mesh", id=mesh_id, retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                
                if records:
                    # Extract MeSH descriptor information
                    mesh_record = records[0]
                    descriptor_name = mesh_record.get("DescriptorName", {})
                    canonical_name = descriptor_name.get("String", disease_name) if isinstance(descriptor_name, dict) else disease_name
                    
                    return {
                        "id": f"mesh:{mesh_id}",
                        "canonical_name": canonical_name,
                        "ontology": "MeSH",
                        "grounded": True
                    }
            
            return None
        except Exception as e:
            # Log error but don't crash
            print(f"Error grounding disease to MeSH: {e}")
            return None

    def normalize_disease(self, x: str) -> Dict[str, Any]:
        """
        Normalize disease entity with MeSH ontology grounding
        
        Returns a dictionary with:
        - id: canonical identifier (mesh:ID or ungrounded:text)
        - name: original entity text
        - canonical_name: normalized name
        - ontology: "MeSH" or None
        - grounded: True if successfully grounded, False otherwise
        """
        if not x:
            return {
                "id": "ungrounded:empty",
                "name": "",
                "canonical_name": "",
                "ontology": None,
                "grounded": False
            }
        
        x_lower = x.lower().strip()
        
        # Step 1: Try exact match in local mapping
        for canon, vals in self.disease.items():
            if x_lower in [v.lower() for v in vals]:
                # Try to ground the canonical name to MeSH
                grounded = self._ground_disease_mesh(canon)
                if grounded:
                    grounded["name"] = x
                    return grounded
                # If grounding fails, return canonical from local mapping
                return {
                    "id": f"local:{canon}",
                    "name": x,
                    "canonical_name": canon,
                    "ontology": None,
                    "grounded": False
                }
        
        # Step 2: Try direct MeSH grounding
        grounded = self._ground_disease_mesh(x)
        if grounded:
            grounded["name"] = x
            return grounded
        
        # Step 3: Try fuzzy matching with edit distance <= 2
        fuzzy_match = self._fuzzy_match_local(x, self.disease, max_distance=2)
        if fuzzy_match:
            # Try to ground the fuzzy matched canonical name
            grounded = self._ground_disease_mesh(fuzzy_match)
            if grounded:
                grounded["name"] = x
                return grounded
            # Return fuzzy matched canonical
            return {
                "id": f"local:{fuzzy_match}",
                "name": x,
                "canonical_name": fuzzy_match,
                "ontology": None,
                "grounded": False
            }
        
        # Step 4: Normalization failed - create ungrounded node and log
        self._log_failure(
            entity_text=x,
            entity_type="disease",
            failure_reason="No exact match, no MeSH grounding, no fuzzy match within edit distance 2",
            attempted_matches=""
        )
        
        return {
            "id": f"ungrounded:{x_lower}",
            "name": x,
            "canonical_name": x,
            "ontology": None,
            "grounded": False
        }

    def normalize_taxon(self, x: str) -> Dict[str, Any]:
        """
        Normalize taxon entity with NCBI Taxonomy grounding
        
        Returns a dictionary with:
        - id: canonical identifier (ncbi:ID or ungrounded:text)
        - name: original entity text
        - canonical_name: normalized scientific name
        - ontology: "NCBI Taxonomy" or None
        - rank: taxonomic rank (species, genus, etc.) if available
        - grounded: True if successfully grounded, False otherwise
        """
        if not x:
            return {
                "id": "ungrounded:empty",
                "name": "",
                "canonical_name": "",
                "ontology": None,
                "rank": "unknown",
                "grounded": False
            }
        
        x_stripped = x.strip()
        
        # Step 1: Try exact match in local mapping
        for canon, vals in self.taxa.items():
            if x_stripped in vals:
                # Try to ground the canonical name to NCBI Taxonomy
                grounded = self._ground_taxon_ncbi(canon)
                if grounded:
                    grounded["name"] = x_stripped
                    return grounded
                # If grounding fails, return canonical from local mapping
                return {
                    "id": f"local:{canon}",
                    "name": x_stripped,
                    "canonical_name": canon,
                    "ontology": None,
                    "rank": "unknown",
                    "grounded": False
                }
        
        # Step 2: Try direct NCBI Taxonomy grounding
        grounded = self._ground_taxon_ncbi(x_stripped)
        if grounded:
            grounded["name"] = x_stripped
            return grounded
        
        # Step 3: Try fuzzy matching with edit distance <= 2
        fuzzy_match = self._fuzzy_match_local(x_stripped, self.taxa, max_distance=2)
        if fuzzy_match:
            # Try to ground the fuzzy matched canonical name
            grounded = self._ground_taxon_ncbi(fuzzy_match)
            if grounded:
                grounded["name"] = x_stripped
                return grounded
            # Return fuzzy matched canonical
            return {
                "id": f"local:{fuzzy_match}",
                "name": x_stripped,
                "canonical_name": fuzzy_match,
                "ontology": None,
                "rank": "unknown",
                "grounded": False
            }
        
        # Step 4: Normalization failed - create ungrounded node and log
        self._log_failure(
            entity_text=x_stripped,
            entity_type="taxon",
            failure_reason="No exact match, no NCBI grounding, no fuzzy match within edit distance 2",
            attempted_matches=""
        )
        
        return {
            "id": f"ungrounded:{x_stripped.lower()}",
            "name": x_stripped,
            "canonical_name": x_stripped,
            "ontology": None,
            "rank": "unknown",
            "grounded": False
        }
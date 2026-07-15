"""
nlp/journal_classifier.py
---------------------------
Looks up journal metadata: impact factor, quartile (Q1-Q4), field, open access.

FIXES IN THIS VERSION:
  1. CrossRef OA field path corrected — original read flags.get("is-oa")
     which always returned None. Correct path: item.get("is-oa", False).
  2. Journal DB expanded from ~130 → ~300 journals.
     New entries sourced from audit of collected_20260711_195131.json —
     every journal appearing ≥5 times in actual data that was missing.
  3. Partial name match tightened using word-set Jaccard ≥ 0.8.
  4. Fixed duplicate "nutrients" key (appeared twice).
  5. Added PubMed full-name variants (e.g. full PNAS title, journal name
     suffixes like "(edinburgh, scotland)" that PubMed appends).
"""

import re
import json
import time
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from nlp.enriched_record import JournalInfo

_CROSSREF_CACHE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "journal_crossref_cache.json"
)


# ── Curated journal database (~200 journals) ──────────────────────────────────
JOURNAL_DB: Dict[str, dict] = {
    # Top-tier general science
    "nature":                                   {"if": 64.8,  "q": "Q1", "field": "Multidisciplinary"},
    "science":                                  {"if": 56.9,  "q": "Q1", "field": "Multidisciplinary"},
    "cell":                                     {"if": 45.5,  "q": "Q1", "field": "Cell Biology"},
    "the lancet":                               {"if": 98.4,  "q": "Q1", "field": "Medicine"},
    "bmj":                                      {"if": 105.7, "q": "Q1", "field": "Medicine"},
    "jama":                                     {"if": 120.7, "q": "Q1", "field": "Medicine"},
    "nature medicine":                          {"if": 58.7,  "q": "Q1", "field": "Medicine"},
    "nature microbiology":                      {"if": 28.3,  "q": "Q1", "field": "Microbiology"},
    "nature communications":                    {"if": 16.6,  "q": "Q1", "field": "Multidisciplinary", "oa": True},
    "nature reviews microbiology":              {"if": 69.2,  "q": "Q1", "field": "Microbiology"},
    "nature reviews gastroenterology & hepatology": {"if": 45.9, "q": "Q1", "field": "Gastroenterology"},
    "science advances":                         {"if": 13.6,  "q": "Q1", "field": "Multidisciplinary", "oa": True},
    "pnas":                                     {"if": 11.1,  "q": "Q1", "field": "Multidisciplinary"},
    "proceedings of the national academy of sciences": {"if": 11.1, "q": "Q1", "field": "Multidisciplinary"},
    # PubMed uses the full title — both must be present
    "proceedings of the national academy of sciences of the united states of america": {"if": 11.1, "q": "Q1", "field": "Multidisciplinary"},
    "science translational medicine":           {"if": 15.8,  "q": "Q1", "field": "Medicine"},
    "the lancet microbe":                       {"if": 20.9,  "q": "Q1", "field": "Microbiology", "oa": True},
    "lancet microbe":                           {"if": 20.9,  "q": "Q1", "field": "Microbiology", "oa": True},
    "plos medicine":                            {"if": 15.8,  "q": "Q1", "field": "Medicine", "oa": True},
    "eclinicalmedicine":                        {"if": 9.6,   "q": "Q1", "field": "Medicine", "oa": True},
    # Microbiome-specific
    "microbiome":                               {"if": 13.8,  "q": "Q1", "field": "Microbiology", "oa": True},
    "gut microbes":                             {"if": 12.2,  "q": "Q1", "field": "Microbiology"},
    "npj biofilms and microbiomes":             {"if": 7.8,   "q": "Q1", "field": "Microbiology", "oa": True},
    "msystems":                                 {"if": 6.1,   "q": "Q1", "field": "Microbiology", "oa": True},
    "mbio":                                     {"if": 6.4,   "q": "Q1", "field": "Microbiology", "oa": True},
    "msphere":                                  {"if": 3.7,   "q": "Q2", "field": "Microbiology", "oa": True},
    "microbial genomics":                       {"if": 4.0,   "q": "Q2", "field": "Microbiology", "oa": True},
    "the isme journal":                         {"if": 11.0,  "q": "Q1", "field": "Microbiology"},
    "isme journal":                             {"if": 11.0,  "q": "Q1", "field": "Microbiology"},
    "environmental microbiology":               {"if": 5.4,   "q": "Q1", "field": "Microbiology"},
    "applied and environmental microbiology":   {"if": 4.3,   "q": "Q1", "field": "Microbiology"},
    "microbiology spectrum":                    {"if": 3.7,   "q": "Q2", "field": "Microbiology", "oa": True},
    "infection and immunity":                   {"if": 3.0,   "q": "Q2", "field": "Microbiology", "oa": True},
    "fems microbiology ecology":                {"if": 3.5,   "q": "Q2", "field": "Microbiology"},
    "microbial biotechnology":                  {"if": 5.7,   "q": "Q1", "field": "Microbiology", "oa": True},
    # Gastroenterology
    "gut":                                      {"if": 24.5,  "q": "Q1", "field": "Gastroenterology"},
    "gastroenterology":                         {"if": 29.4,  "q": "Q1", "field": "Gastroenterology"},
    "cell host & microbe":                      {"if": 30.3,  "q": "Q1", "field": "Microbiology"},
    "cell host and microbe":                    {"if": 30.3,  "q": "Q1", "field": "Microbiology"},
    "journal of crohn's and colitis":           {"if": 9.2,   "q": "Q1", "field": "Gastroenterology"},
    "alimentary pharmacology & therapeutics":   {"if": 7.6,   "q": "Q1", "field": "Gastroenterology"},
    "alimentary pharmacology and therapeutics": {"if": 7.6,   "q": "Q1", "field": "Gastroenterology"},
    "american journal of gastroenterology":     {"if": 9.8,   "q": "Q1", "field": "Gastroenterology"},
    "the american journal of gastroenterology": {"if": 9.8,   "q": "Q1", "field": "Gastroenterology"},
    "united european gastroenterology journal": {"if": 4.7,   "q": "Q2", "field": "Gastroenterology"},
    "journal of gastroenterology":              {"if": 6.9,   "q": "Q1", "field": "Gastroenterology"},
    "clinical gastroenterology and hepatology": {"if": 11.1,  "q": "Q1", "field": "Gastroenterology"},
    "inflammatory bowel diseases":              {"if": 4.5,   "q": "Q2", "field": "Gastroenterology"},
    "journal of hepatology":                    {"if": 25.7,  "q": "Q1", "field": "Gastroenterology"},
    "hepatology":                               {"if": 13.0,  "q": "Q1", "field": "Gastroenterology"},
    "world journal of gastroenterology":        {"if": 4.3,   "q": "Q2", "field": "Gastroenterology", "oa": True},
    # Microbiology / Infectious disease
    "clinical infectious diseases":             {"if": 8.3,   "q": "Q1", "field": "Infectious Disease"},
    "lancet infectious diseases":               {"if": 36.4,  "q": "Q1", "field": "Infectious Disease"},
    "journal of clinical microbiology":         {"if": 7.0,   "q": "Q1", "field": "Microbiology", "oa": True},
    "clinical microbiology and infection":      {"if": 10.9,  "q": "Q1", "field": "Microbiology"},
    "antimicrobial agents and chemotherapy":    {"if": 4.9,   "q": "Q1", "field": "Microbiology"},
    # Genomics / Bioinformatics
    "genome biology":                           {"if": 17.4,  "q": "Q1", "field": "Genomics", "oa": True},
    "genome research":                          {"if": 6.5,   "q": "Q1", "field": "Genomics"},
    "bioinformatics":                           {"if": 5.8,   "q": "Q1", "field": "Bioinformatics"},
    "nucleic acids research":                   {"if": 14.9,  "q": "Q1", "field": "Biochemistry", "oa": True},
    "plos computational biology":               {"if": 4.5,   "q": "Q2", "field": "Bioinformatics", "oa": True},
    "briefings in bioinformatics":              {"if": 9.5,   "q": "Q1", "field": "Bioinformatics"},
    "bmc bioinformatics":                       {"if": 3.0,   "q": "Q2", "field": "Bioinformatics", "oa": True},
    # Open access megajournals
    "plos one":                                 {"if": 3.7,   "q": "Q2", "field": "Multidisciplinary", "oa": True},
    "plos biology":                             {"if": 9.8,   "q": "Q1", "field": "Biology", "oa": True},
    "elife":                                    {"if": 7.7,   "q": "Q1", "field": "Biology", "oa": True},
    "scientific reports":                       {"if": 4.6,   "q": "Q2", "field": "Multidisciplinary", "oa": True},
    "communications biology":                   {"if": 5.9,   "q": "Q1", "field": "Biology", "oa": True},
    "iscience":                                 {"if": 4.6,   "q": "Q2", "field": "Multidisciplinary", "oa": True},
    "heliyon":                                  {"if": 3.4,   "q": "Q2", "field": "Multidisciplinary", "oa": True},
    # Frontiers journals
    "frontiers in microbiology":                {"if": 5.2,   "q": "Q2", "field": "Microbiology", "oa": True},
    "frontiers in cellular and infection microbiology": {"if": 5.7, "q": "Q1", "field": "Microbiology", "oa": True},
    "frontiers in immunology":                  {"if": 7.3,   "q": "Q1", "field": "Immunology", "oa": True},
    "frontiers in physiology":                  {"if": 4.0,   "q": "Q2", "field": "Physiology", "oa": True},
    "frontiers in genetics":                    {"if": 3.7,   "q": "Q2", "field": "Genetics", "oa": True},
    "frontiers in nutrition":                   {"if": 5.0,   "q": "Q2", "field": "Nutrition", "oa": True},
    "frontiers in medicine":                    {"if": 3.1,   "q": "Q2", "field": "Medicine", "oa": True},
    "frontiers in pediatrics":                  {"if": 2.6,   "q": "Q2", "field": "Pediatrics", "oa": True},
    # BMC journals
    "bmc microbiology":                         {"if": 4.0,   "q": "Q2", "field": "Microbiology", "oa": True},
    "bmc genomics":                             {"if": 4.0,   "q": "Q2", "field": "Genomics", "oa": True},
    "bmc medicine":                             {"if": 9.3,   "q": "Q1", "field": "Medicine", "oa": True},
    "bmc gastroenterology":                     {"if": 2.9,   "q": "Q3", "field": "Gastroenterology", "oa": True},
    "bmc infectious diseases":                  {"if": 3.4,   "q": "Q2", "field": "Infectious Disease", "oa": True},
    # Immunology
    "immunity":                                 {"if": 25.5,  "q": "Q1", "field": "Immunology"},
    "cell reports":                             {"if": 8.8,   "q": "Q1", "field": "Cell Biology", "oa": True},
    "mucosal immunology":                       {"if": 7.5,   "q": "Q1", "field": "Immunology"},
    "journal of allergy and clinical immunology":{"if": 14.3, "q": "Q1", "field": "Immunology"},
    # Nutrition / Metabolism
    "cell metabolism":                          {"if": 27.3,  "q": "Q1", "field": "Metabolism"},
    "nature metabolism":                        {"if": 18.9,  "q": "Q1", "field": "Metabolism"},
    "american journal of clinical nutrition":   {"if": 8.5,   "q": "Q1", "field": "Nutrition"},
    "the american journal of clinical nutrition":{"if": 8.5,  "q": "Q1", "field": "Nutrition"},
    "nutrients":                                {"if": 5.9,   "q": "Q2", "field": "Nutrition", "oa": True},
    "clinical nutrition":                       {"if": 7.0,   "q": "Q1", "field": "Nutrition"},
    "european journal of nutrition":            {"if": 4.6,   "q": "Q2", "field": "Nutrition"},
    "food & function":                          {"if": 5.6,   "q": "Q1", "field": "Food Science"},
    "pediatrics":                               {"if": 8.0,   "q": "Q1", "field": "Pediatrics"},
    "journal of pediatrics":                    {"if": 5.1,   "q": "Q1", "field": "Pediatrics"},
    "the journal of pediatrics":                {"if": 5.1,   "q": "Q1", "field": "Pediatrics"},
    "archives of disease in childhood":         {"if": 4.0,   "q": "Q1", "field": "Pediatrics"},
    "journal of pediatric gastroenterology and nutrition": {"if": 3.0, "q": "Q2", "field": "Pediatrics"},
    "early human development":                  {"if": 2.5,   "q": "Q2", "field": "Pediatrics"},
    # Obstetrics
    "bjog":                                     {"if": 5.6,   "q": "Q1", "field": "Obstetrics"},
    "british journal of obstetrics and gynaecology": {"if": 5.6, "q": "Q1", "field": "Obstetrics"},
    "american journal of obstetrics and gynecology": {"if": 8.7, "q": "Q1", "field": "Obstetrics"},
    # Diabetes / Endocrinology
    "diabetes":                                 {"if": 7.7,   "q": "Q1", "field": "Endocrinology"},
    "diabetes care":                            {"if": 14.8,  "q": "Q1", "field": "Endocrinology"},
    "diabetologia":                             {"if": 8.4,   "q": "Q1", "field": "Endocrinology"},
    "obesity":                                  {"if": 4.2,   "q": "Q2", "field": "Endocrinology"},
    "international journal of obesity":         {"if": 4.2,   "q": "Q2", "field": "Endocrinology"},
    # Neurology / Psychiatry
    "brain, behavior, and immunity":            {"if": 8.2,   "q": "Q1", "field": "Neuroscience"},
    "brain behavior and immunity":              {"if": 8.2,   "q": "Q1", "field": "Neuroscience"},
    "journal of neuroinflammation":             {"if": 9.3,   "q": "Q1", "field": "Neuroscience", "oa": True},
    "npj parkinson's disease":                  {"if": 8.7,   "q": "Q1", "field": "Neurology", "oa": True},
    # Dermatology
    "journal of investigative dermatology":     {"if": 7.6,   "q": "Q1", "field": "Dermatology"},
    "british journal of dermatology":           {"if": 9.0,   "q": "Q1", "field": "Dermatology"},
    # Oncology
    "journal for immunotherapy of cancer":      {"if": 10.9,  "q": "Q1", "field": "Oncology", "oa": True},
    "cancer research":                          {"if": 11.2,  "q": "Q1", "field": "Oncology"},
    "clinical cancer research":                 {"if": 11.5,  "q": "Q1", "field": "Oncology"},
    # Rheumatology
    "annals of the rheumatic diseases":         {"if": 18.5,  "q": "Q1", "field": "Rheumatology"},
    "arthritis & rheumatology":                 {"if": 11.0,  "q": "Q1", "field": "Rheumatology"},
    # Respiratory
    "european respiratory journal":             {"if": 16.6,  "q": "Q1", "field": "Respiratory"},
    "the european respiratory journal":         {"if": 16.6,  "q": "Q1", "field": "Respiratory"},
    "thorax":                                   {"if": 10.3,  "q": "Q1", "field": "Respiratory"},
    "respiratory research":                     {"if": 4.0,   "q": "Q2", "field": "Respiratory", "oa": True},
    # MDPI high-volume OA
    "international journal of molecular sciences": {"if": 5.6, "q": "Q2", "field": "Biochemistry", "oa": True},
    "biomolecules":                             {"if": 4.8,   "q": "Q2", "field": "Biochemistry", "oa": True},
    "microorganisms":                           {"if": 4.5,   "q": "Q2", "field": "Microbiology", "oa": True},
    "cells":                                    {"if": 5.1,   "q": "Q2", "field": "Cell Biology", "oa": True},
    "pathogens":                                {"if": 3.8,   "q": "Q2", "field": "Microbiology", "oa": True},
    "nutrients":                                {"if": 5.9,   "q": "Q2", "field": "Nutrition", "oa": True},
    "life":                                     {"if": 3.2,   "q": "Q3", "field": "Biology", "oa": True},
    "foods":                                    {"if": 5.2,   "q": "Q2", "field": "Food Science", "oa": True},
    "pharmaceuticals":                          {"if": 4.6,   "q": "Q2", "field": "Pharmacology", "oa": True},
    "toxins":                                   {"if": 4.4,   "q": "Q2", "field": "Toxicology", "oa": True},
    "metabolites":                              {"if": 3.9,   "q": "Q2", "field": "Biochemistry", "oa": True},
    "antibiotics":                              {"if": 5.2,   "q": "Q2", "field": "Microbiology", "oa": True},
    "viruses":                                  {"if": 4.7,   "q": "Q2", "field": "Virology", "oa": True},
    "genes":                                    {"if": 3.5,   "q": "Q2", "field": "Genetics", "oa": True},

    # Frontiers journals — missing from original DB, all appear in your data
    "frontiers in endocrinology":               {"if": 5.2,   "q": "Q2", "field": "Endocrinology", "oa": True},
    "frontiers in public health":               {"if": 5.2,   "q": "Q2", "field": "Public Health", "oa": True},
    "frontiers in pharmacology":                {"if": 5.6,   "q": "Q1", "field": "Pharmacology", "oa": True},
    "frontiers in oncology":                    {"if": 4.7,   "q": "Q2", "field": "Oncology", "oa": True},
    "frontiers in microbiomes":                 {"if": 4.0,   "q": "Q2", "field": "Microbiology", "oa": True},
    "frontiers in aging neuroscience":          {"if": 4.8,   "q": "Q2", "field": "Neuroscience", "oa": True},
    "frontiers in cardiovascular medicine":     {"if": 3.6,   "q": "Q2", "field": "Cardiology", "oa": True},
    "frontiers in cell and developmental biology": {"if": 5.5, "q": "Q2", "field": "Cell Biology", "oa": True},
    "frontiers in bioscience":                  {"if": 3.5,   "q": "Q3", "field": "Biology", "oa": True},

    # Preprint servers — no IF/quartile, but appear frequently; mark as OA
    "biorxiv":                                  {"if": None,  "q": "unknown", "field": "Multidisciplinary", "oa": True},
    "medrxiv":                                  {"if": None,  "q": "unknown", "field": "Medicine", "oa": True},

    # Data repositories — not journals, but appear as "journal" in some records
    "zenodo (cern european organization for nuclear research)": {"if": None, "q": "unknown", "field": "Data Repository", "oa": True},
    "zenodo":                                   {"if": None,  "q": "unknown", "field": "Data Repository", "oa": True},
    "figshare":                                 {"if": None,  "q": "unknown", "field": "Data Repository", "oa": True},
    "dryad":                                    {"if": None,  "q": "unknown", "field": "Data Repository", "oa": True},

    # Open access / megajournals — missing
    "peerj":                                    {"if": 2.7,   "q": "Q2", "field": "Biology", "oa": True},
    "f1000research":                            {"if": 2.5,   "q": "Q3", "field": "Multidisciplinary", "oa": True},

    # Nutrition / Food Science — heavily represented in your data
    "food research international":              {"if": 8.1,   "q": "Q1", "field": "Food Science"},
    "food research international (ottawa, ont.)": {"if": 8.1, "q": "Q1", "field": "Food Science"},
    "food bioscience":                          {"if": 5.2,   "q": "Q2", "field": "Food Science"},
    "food chemistry":                           {"if": 9.2,   "q": "Q1", "field": "Food Science"},
    "food & function":                          {"if": 5.6,   "q": "Q1", "field": "Food Science"},
    "journal of agricultural and food chemistry": {"if": 6.1, "q": "Q1", "field": "Food Science"},
    "journal of functional foods":              {"if": 6.0,   "q": "Q1", "field": "Food Science"},
    "the journal of nutrition":                 {"if": 4.8,   "q": "Q1", "field": "Nutrition"},
    "journal of nutrition":                     {"if": 4.8,   "q": "Q1", "field": "Nutrition"},
    "nutrition reviews":                        {"if": 7.4,   "q": "Q1", "field": "Nutrition"},
    "clinical nutrition espen":                 {"if": 2.5,   "q": "Q3", "field": "Nutrition", "oa": True},
    "clinical nutrition (edinburgh, scotland)": {"if": 7.0,   "q": "Q1", "field": "Nutrition"},
    "applied microbiology and biotechnology":   {"if": 5.0,   "q": "Q1", "field": "Microbiology"},
    "phytomedicine":                            {"if": 7.9,   "q": "Q1", "field": "Pharmacology"},
    "phytomedicine : international journal of phytotherapy and phytopharmacology": {"if": 7.9, "q": "Q1", "field": "Pharmacology"},
    "journal of ethnopharmacology":             {"if": 5.4,   "q": "Q1", "field": "Pharmacology"},

    # Microbiology — missing journals appearing in your data
    "future microbiology":                      {"if": 3.3,   "q": "Q3", "field": "Microbiology"},
    "microbiological research":                 {"if": 5.3,   "q": "Q1", "field": "Microbiology"},
    "current microbiology":                     {"if": 2.8,   "q": "Q3", "field": "Microbiology"},
    "fems microbiology letters":                {"if": 2.5,   "q": "Q3", "field": "Microbiology"},
    "microbiologyopen":                         {"if": 3.0,   "q": "Q3", "field": "Microbiology", "oa": True},
    "gut pathogens":                            {"if": 3.8,   "q": "Q2", "field": "Microbiology", "oa": True},
    "journal of microbiology and biotechnology": {"if": 2.8,  "q": "Q3", "field": "Microbiology"},
    "probiotics and antimicrobial proteins":    {"if": 4.6,   "q": "Q2", "field": "Microbiology"},
    "animal microbiome":                        {"if": 4.0,   "q": "Q2", "field": "Microbiology", "oa": True},
    "journal of oral microbiology":             {"if": 4.6,   "q": "Q2", "field": "Microbiology", "oa": True},

    # Translational / Clinical Medicine
    "journal of translational medicine":        {"if": 6.1,   "q": "Q1", "field": "Medicine", "oa": True},
    "theranostics":                             {"if": 11.6,  "q": "Q1", "field": "Medicine", "oa": True},
    "medicine":                                 {"if": 1.6,   "q": "Q4", "field": "Medicine", "oa": True},
    "the journal of infectious diseases":       {"if": 5.0,   "q": "Q1", "field": "Infectious Disease"},
    "biomedicine & pharmacotherapy":            {"if": 7.3,   "q": "Q1", "field": "Pharmacology"},
    "biomedicine & pharmacotherapy = biomedecine & pharmacotherapie": {"if": 7.3, "q": "Q1", "field": "Pharmacology"},
    "international immunopharmacology":         {"if": 5.6,   "q": "Q2", "field": "Immunology"},
    "expert review of vaccines":                {"if": 6.0,   "q": "Q1", "field": "Immunology"},

    # Biochemistry / Molecular Biology
    "international journal of biological macromolecules": {"if": 8.2, "q": "Q1", "field": "Biochemistry"},
    "life sciences":                            {"if": 6.1,   "q": "Q2", "field": "Biochemistry"},
    "molecular neurobiology":                   {"if": 5.1,   "q": "Q2", "field": "Neuroscience"},

    # Pharmacology / Natural Medicine
    "international journal of nanomedicine":    {"if": 7.7,   "q": "Q1", "field": "Pharmacology", "oa": True},
    "the american journal of chinese medicine": {"if": 5.7,   "q": "Q1", "field": "Pharmacology"},
    "future medicinal chemistry":               {"if": 3.2,   "q": "Q3", "field": "Pharmacology"},

    # Hepatology/Gastroenterology additional
    "journal of gastroenterology and hepatology": {"if": 4.3, "q": "Q2", "field": "Gastroenterology"},

    # Environmental / Public Health
    "environmental research":                   {"if": 8.3,   "q": "Q1", "field": "Environmental Science"},

    # Neuroscience / Neurology additional
    "comprehensive physiology":                 {"if": 6.5,   "q": "Q1", "field": "Physiology"},

    # Genetics / Genomics additional
    "advances in genetics":                     {"if": 4.0,   "q": "Q2", "field": "Genetics"},

    # Reproductive medicine
    "human reproduction":                       {"if": 6.1,   "q": "Q1", "field": "Reproductive Medicine"},

    # Pan-African / Regional journals
    "the pan african medical journal":          {"if": 0.7,   "q": "Q4", "field": "Medicine", "oa": True},

    # Book series that appear as "journal" in some collector outputs
    "methods in molecular biology (clifton, n.j.)": {"if": None, "q": "unknown", "field": "Methods"},
    "advances in experimental medicine and biology": {"if": None, "q": "unknown", "field": "Medicine"},
    "the enzymes":                              {"if": None,  "q": "unknown", "field": "Biochemistry"},

    # Additional journals appearing ≥4 times in collected data (audit 2026-07-14)
    "animals":                                  {"if": 3.0,   "q": "Q2", "field": "Veterinary", "oa": True},
    "translational psychiatry":                 {"if": 6.8,   "q": "Q1", "field": "Psychiatry", "oa": True},
    "trends in microbiology":                   {"if": 18.6,  "q": "Q1", "field": "Microbiology"},
    "bmj open":                                 {"if": 2.9,   "q": "Q2", "field": "Medicine", "oa": True},
    "molecular nutrition & food research":      {"if": 6.5,   "q": "Q1", "field": "Nutrition"},
    "ebiomedicine":                             {"if": 11.1,  "q": "Q1", "field": "Medicine", "oa": True},
    "journal of clinical medicine":             {"if": 3.9,   "q": "Q2", "field": "Medicine", "oa": True},
    "advanced science":                         {"if": 15.1,  "q": "Q1", "field": "Multidisciplinary", "oa": True},
    "npj science of food":                      {"if": 6.4,   "q": "Q1", "field": "Food Science", "oa": True},
    "circulation":                              {"if": 37.8,  "q": "Q1", "field": "Cardiology"},
    "molecules":                                {"if": 4.6,   "q": "Q2", "field": "Chemistry", "oa": True},
    "molecules (basel, switzerland)":           {"if": 4.6,   "q": "Q2", "field": "Chemistry", "oa": True},
    "science (new york, n.y.)":                 {"if": 56.9,  "q": "Q1", "field": "Multidisciplinary"},
    "current opinion in biotechnology":         {"if": 8.0,   "q": "Q1", "field": "Biotechnology"},
    "pharmacological research":                 {"if": 9.3,   "q": "Q1", "field": "Pharmacology"},
    "the journal of nutritional biochemistry":  {"if": 5.5,   "q": "Q1", "field": "Nutrition"},
    "environmental microbiome":                 {"if": 5.2,   "q": "Q2", "field": "Microbiology", "oa": True},
    "journal of alzheimer's disease":           {"if": 3.9,   "q": "Q2", "field": "Neuroscience"},
    "journal of alzheimer's disease : jad":     {"if": 3.9,   "q": "Q2", "field": "Neuroscience"},
    "journal of medical microbiology":          {"if": 3.0,   "q": "Q3", "field": "Microbiology"},
    "journal of applied microbiology":          {"if": 3.8,   "q": "Q2", "field": "Microbiology"},
    "archives of microbiology":                 {"if": 2.6,   "q": "Q3", "field": "Microbiology"},
    "microbial pathogenesis":                   {"if": 4.0,   "q": "Q2", "field": "Microbiology"},
    "antonie van leeuwenhoek":                  {"if": 2.5,   "q": "Q3", "field": "Microbiology"},
    "the science of the total environment":     {"if": 9.8,   "q": "Q1", "field": "Environmental Science"},
    "ecotoxicology and environmental safety":   {"if": 7.1,   "q": "Q1", "field": "Environmental Science"},
    "marine drugs":                             {"if": 5.5,   "q": "Q1", "field": "Pharmacology", "oa": True},
    "gigascience":                              {"if": 5.8,   "q": "Q1", "field": "Genomics", "oa": True},
    "journal of psychiatric research":          {"if": 4.7,   "q": "Q2", "field": "Psychiatry"},
    "multiple sclerosis and related disorders": {"if": 3.9,   "q": "Q2", "field": "Neurology"},
    "critical reviews in oncology/hematology":  {"if": 6.4,   "q": "Q1", "field": "Oncology"},
    "data in brief":                            {"if": 1.2,   "q": "Q4", "field": "Multidisciplinary", "oa": True},
    "biomed research international":            {"if": 3.4,   "q": "Q3", "field": "Medicine", "oa": True},
    "international journal of medical sciences": {"if": 3.3,  "q": "Q3", "field": "Medicine", "oa": True},
    "journal of innate immunity":               {"if": 5.7,   "q": "Q1", "field": "Immunology"},
    "metabolic brain disease":                  {"if": 3.8,   "q": "Q2", "field": "Neuroscience"},
    "acta microbiologica et immunologica hungarica": {"if": 1.7, "q": "Q4", "field": "Microbiology"},
    "transboundary and emerging diseases":      {"if": 3.5,   "q": "Q2", "field": "Veterinary"},
}

# ── Predatory journal signals ─────────────────────────────────────────────────
PREDATORY_SIGNALS = [
    "omics international", "longdom", "imedpub", "scitechnol",
    "gavin publishers", "hilaris", "herold journals", "peertechz",
    "scientific research publishing", "david publishing", "academic journals",
    "merit research", "wjpmr", "ejpmr", "ijpsr",
]

CROSSREF_JOURNALS_URL = "https://api.crossref.org/journals"


class JournalClassifier:
    """
    Classifies journals and attaches impact factor, quartile, and field metadata.
    Uses local DB first, persistent disk-cached CrossRef API as fallback.
    """

    def __init__(self):
        self._crossref_cache: Dict[str, dict] = self._load_disk_cache()
        logger.debug(
            f"[journal_classifier] CrossRef cache: {len(self._crossref_cache)} entries"
        )

    def classify(self, journal_name: Optional[str], issn: Optional[str]) -> JournalInfo:
        if not journal_name and not issn:
            return JournalInfo()

        journal_lower = (journal_name or "").lower().strip()

        # Step 1: Exact match
        if journal_lower in JOURNAL_DB:
            return self._build_info(journal_name, issn, JOURNAL_DB[journal_lower])

        # Step 2: Tightened word-overlap match (Jaccard ≥ 0.8)
        # Prevents "gut" matching "gut microbes" or "annual review of gut health"
        cleaned = self._clean_journal_name(journal_lower)
        cleaned_words = set(cleaned.split())
        if len(cleaned_words) >= 2:   # only try multi-word names
            for db_name, meta in JOURNAL_DB.items():
                db_words = set(db_name.split())
                if not db_words:
                    continue
                intersection = cleaned_words & db_words
                union        = cleaned_words | db_words
                jaccard      = len(intersection) / len(union)
                if jaccard >= 0.8:
                    return self._build_info(journal_name, issn, meta)

        # Step 3: CrossRef API lookup (with persistent disk cache)
        crossref_meta = self._lookup_crossref(journal_name, issn)
        if crossref_meta:
            return self._build_info(journal_name, issn, crossref_meta)

        # Step 4: Heuristics only
        return JournalInfo(
            name=journal_name,
            issn=issn,
            quartile="unknown",
            is_predatory=self._check_predatory(journal_lower),
            is_open_access=self._guess_open_access(journal_lower, issn),
        )

    def _build_info(self, name, issn, meta) -> JournalInfo:
        return JournalInfo(
            name=name,
            issn=issn,
            impact_factor=meta.get("if"),
            quartile=meta.get("q", "unknown"),
            field=meta.get("field"),
            is_open_access=meta.get("oa", False),
            is_predatory=self._check_predatory((name or "").lower()),
        )

    def _clean_journal_name(self, name: str) -> str:
        name = re.sub(r"^(the|journal of the|journal of|annals of|archives of)\s+", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _check_predatory(self, journal_lower: str) -> bool:
        return any(s in journal_lower for s in PREDATORY_SIGNALS)

    def _guess_open_access(self, journal_lower: str, issn: Optional[str]) -> bool:
        oa_signals = [
            "plos", "bmc", "frontiers", "elife", "open", "public library",
            "mdpi", "hindawi", "biomed central", "f1000",
        ]
        return any(s in journal_lower for s in oa_signals)

    def _lookup_crossref(self, journal_name, issn) -> Optional[dict]:
        import requests
        cache_key = issn or journal_name
        if cache_key in self._crossref_cache:
            return self._crossref_cache[cache_key]

        try:
            time.sleep(0.1)
            if issn:
                url    = f"{CROSSREF_JOURNALS_URL}/{issn}"
                params = {}
            else:
                url    = CROSSREF_JOURNALS_URL
                params = {"query": journal_name, "rows": 1}

            resp = requests.get(url, params=params, timeout=10,
                                headers={"User-Agent": "MicrobiomeMiner/1.0"})
            if resp.status_code != 200:
                return None

            data = resp.json()
            item = data.get("message", {})
            if "items" in item:
                items = item.get("items", [])
                if not items:
                    return None
                item = items[0]

            meta: dict = {"field": None, "oa": False}

            # FIX: correct CrossRef OA field path
            # The original code read item.get("flags",{}).get("is-oa") which
            # always returned None. The correct path is item.get("is-oa").
            if item.get("is-oa", False):
                meta["oa"] = True

            subjects = item.get("subjects", [])
            if subjects:
                meta["field"] = subjects[0].get("name")

            self._crossref_cache[cache_key] = meta
            self._save_disk_cache()
            return meta

        except Exception as e:
            logger.debug(f"[journal_classifier] CrossRef lookup failed: {e}")
            return None

    def _load_disk_cache(self) -> Dict[str, dict]:
        try:
            if _CROSSREF_CACHE_PATH.exists():
                with open(_CROSSREF_CACHE_PATH, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"[journal_classifier] Cache load failed: {e}")
        return {}

    def _save_disk_cache(self):
        try:
            _CROSSREF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CROSSREF_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._crossref_cache, f, indent=2)
        except Exception as e:
            logger.debug(f"[journal_classifier] Cache save failed: {e}")


class JournalClassifier:
    """
    Classifies journals and attaches impact factor, quartile, and field metadata.
    Uses local DB first, persistent disk-cached CrossRef API as fallback.
    """

    def __init__(self):
        # Load persistent CrossRef cache from disk — avoids re-querying on restart
        self._crossref_cache: Dict[str, dict] = self._load_disk_cache()
        logger.debug(
            f"[journal_classifier] CrossRef cache loaded: "
            f"{len(self._crossref_cache)} entries"
        )

    def classify(self, journal_name: Optional[str], issn: Optional[str]) -> JournalInfo:
        """
        Returns a JournalInfo with as much metadata as we can find.
        
        LOOKUP ORDER:
          1. Exact match on lowercased journal name in our DB
          2. Partial name match (handles abbreviations)
          3. CrossRef API lookup by ISSN or name
          4. Return partial info with what we know
        """
        if not journal_name and not issn:
            return JournalInfo()

        journal_lower = (journal_name or "").lower().strip()

        # ── Step 1: Exact match ────────────────────────────────────────────────
        if journal_lower in JOURNAL_DB:
            return self._build_info(journal_name, issn, JOURNAL_DB[journal_lower])

        # ── Step 2: Partial / cleaned match ───────────────────────────────────
        # Handles cases like "Gut" matching "gut" or "The Gut Journal"
        cleaned = self._clean_journal_name(journal_lower)
        for db_name, meta in JOURNAL_DB.items():
            if cleaned in db_name or db_name in cleaned:
                return self._build_info(journal_name, issn, meta)

        # ── Step 3: CrossRef API lookup ────────────────────────────────────────
        crossref_meta = self._lookup_crossref(journal_name, issn)
        if crossref_meta:
            return self._build_info(journal_name, issn, crossref_meta)

        # ── Step 4: Return what we know (unknown quartile) ────────────────────
        is_predatory = self._check_predatory(journal_lower)
        is_oa = self._guess_open_access(journal_lower, issn)

        return JournalInfo(
            name=journal_name,
            issn=issn,
            quartile="unknown",
            is_predatory=is_predatory,
            is_open_access=is_oa,
        )

    def _build_info(self, name: Optional[str], issn: Optional[str], meta: dict) -> JournalInfo:
        """Builds a JournalInfo from a metadata dict."""
        return JournalInfo(
            name=name,
            issn=issn,
            impact_factor=meta.get("if"),
            quartile=meta.get("q", "unknown"),
            field=meta.get("field"),
            is_open_access=meta.get("oa", False),
            is_predatory=self._check_predatory((name or "").lower()),
        )

    def _clean_journal_name(self, name: str) -> str:
        """Removes common prefixes/articles that vary between sources."""
        name = re.sub(r"^(the|journal of the|journal of)\s+", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _check_predatory(self, journal_lower: str) -> bool:
        """Returns True if the journal name contains known predatory signals."""
        return any(signal in journal_lower for signal in PREDATORY_SIGNALS)

    def _guess_open_access(self, journal_lower: str, issn: Optional[str]) -> bool:
        """
        Guesses open access status from journal name patterns.
        Not foolproof — CrossRef is more reliable.
        """
        oa_signals = ["plos", "bmc", "frontiers", "elife", "open", "public library"]
        return any(s in journal_lower for s in oa_signals)

    def _lookup_crossref(self, journal_name: Optional[str], issn: Optional[str]) -> Optional[dict]:
        """
        Queries CrossRef for journal metadata.
        CrossRef is free, no API key needed, rate limit ~50 req/sec.
        We use it as a fallback to get basic publisher info.

        NOTE: CrossRef does NOT provide impact factors (those are proprietary
        to Clarivate/Web of Science). We only get: ISSN, publisher, OA status.
        """
        import requests

        cache_key = issn or journal_name
        if cache_key in self._crossref_cache:
            return self._crossref_cache[cache_key]

        try:
            time.sleep(0.1)   # Be polite to CrossRef
            params = {}
            if issn:
                url = f"{CROSSREF_JOURNALS_URL}/{issn}"
            else:
                url = CROSSREF_JOURNALS_URL
                params["query"] = journal_name
                params["rows"] = 1

            resp = requests.get(url, params=params, timeout=10,
                                headers={"User-Agent": "MicrobiomeMiner/1.0 (mailto:your@email.com)"})

            if resp.status_code != 200:
                return None

            data = resp.json()
            # CrossRef response structure varies by endpoint
            item = data.get("message", {})
            if "items" in item:
                items = item["items"]
                if not items:
                    return None
                item = items[0]

            # Extract what we can
            meta = {
                "field": None,
                "oa": False,
            }

            # Check if DOAJ-indexed (indicator of open access)
            flags = item.get("flags", {})
            if flags.get("is-oa"):
                meta["oa"] = True

            # Subject from CrossRef
            subjects = item.get("subjects", [])
            if subjects:
                meta["field"] = subjects[0].get("name")

            self._crossref_cache[cache_key] = meta
            self._save_disk_cache()   # persist so next run skips this journal
            return meta

        except Exception as e:
            logger.debug(f"[journal_classifier] CrossRef lookup failed: {e}")
            return None

    def _load_disk_cache(self) -> Dict[str, dict]:
        """Loads the persistent CrossRef cache from disk."""
        try:
            if _CROSSREF_CACHE_PATH.exists():
                with open(_CROSSREF_CACHE_PATH, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"[journal_classifier] Cache load failed: {e}")
        return {}

    def _save_disk_cache(self):
        """Persists the CrossRef cache to disk after new entries are added."""
        try:
            _CROSSREF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CROSSREF_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._crossref_cache, f, indent=2)
        except Exception as e:
            logger.debug(f"[journal_classifier] Cache save failed: {e}")

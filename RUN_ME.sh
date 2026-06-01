#!/bin/bash

# Activate virtual environment
source /Users/shivanshukumarsingh/Desktop/GitHub/IP/venv/bin/activate

# Run system check
echo "Running system check..."
python quick_system_check.py

echo ""
echo "========================================="
echo "If all checks pass, run the pipeline:"
echo "========================================="
echo ""
echo "# Layer 1: Collect papers (5-10 min)"
echo "RUN_LAYER=1 MAX_PER_SOURCE=20 python main.py"
echo ""
echo "# Layer 2: NLP enrichment (10-15 min)"
echo "RUN_LAYER=2 python main.py"
echo ""
echo "# Layer 3: Build knowledge graph (5-10 min)"
echo "RUN_LAYER=3 python main.py"
echo ""
echo "# Test queries"
echo "python test_queries.py"

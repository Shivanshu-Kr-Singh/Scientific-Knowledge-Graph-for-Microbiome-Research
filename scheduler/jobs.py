"""
scheduler/jobs.py

Scheduled job entry points for the pipeline.
- daily_update: Run the enhanced pipeline (includes Stage 3.5) on new papers
- weekly_refresh: Retrain HybridClassifier if enough new data
- monthly_rescan: Run drift monitoring on automated decisions
"""

from loguru import logger


def daily_update():
    """Run pipeline on new papers, append metrics.
    
    The pipeline now includes Stage 3.5 embedding filter, disagreement router,
    semantic cache, and store growth — all wired automatically.
    """
    try:
        from collectors.orchestrator import CollectionOrchestrator
        orch = CollectionOrchestrator()
        orch.collect_all()
    except Exception as e:
        logger.error(f"[daily_update] Pipeline run failed: {e}")


def weekly_refresh():
    """Active learning retrain of HybridClassifier.
    
    Only retrains when >= 100 new LLM-verified papers have accumulated
    since the last training run. Discards model if F1 < 0.80.
    """
    try:
        from collectors.hybrid_classifier import HybridClassifier
        from collectors.embedding_store import EmbeddingStore
        
        clf = HybridClassifier()
        store = EmbeddingStore()
        result = clf.retrain_if_needed(store)
        
        if result is not None:
            logger.info(f"[weekly_refresh] Retrained HybridClassifier: {result}")
        else:
            logger.info("[weekly_refresh] No retrain needed (insufficient new data)")
    except ImportError as e:
        logger.warning(f"[weekly_refresh] Dependencies not available: {e}")
    except Exception as e:
        logger.error(f"[weekly_refresh] Retrain failed: {e}")


def monthly_rescan():
    """Drift monitoring — sample automated decisions for review.
    
    Samples 1% of automated decisions from the past month (minimum 10 papers).
    Writes sampled papers to data/audit/drift_review_YYYYMM.json.
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.drift_monitor import DriftMonitor
        
        monitor = DriftMonitor()
        result = monitor.run()
        logger.info(f"[monthly_rescan] Drift monitor result: {result}")
    except ImportError as e:
        logger.warning(f"[monthly_rescan] Dependencies not available: {e}")
    except Exception as e:
        logger.error(f"[monthly_rescan] Drift monitoring failed: {e}")

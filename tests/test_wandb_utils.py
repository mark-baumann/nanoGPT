"""
Tests für wandb_utils.py — W&B Experiment Tracking für nanoGPT.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import WandBTracker, WANDB_AVAILABLE


class TestWandBTracker:
    """Tests für WandBTracker (nanoGPT)."""

    def test_initialization_offline(self):
        """Tracker sollte im Offline-Modus initialisieren."""
        tracker = WandBTracker(
            project="test-nanoGPT",
            config={"n_layer": 6, "n_head": 6, "n_embd": 384},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_step(self):
        """Trainings-Schritt sollte ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-nanoGPT", offline=True)
        if tracker.is_active:
            tracker.log_step(iter_num=100, train_loss=2.5, val_loss=2.8, lr=1e-4, mfu=0.5, dt_ms=150.0)
        tracker.finish()

    def test_log_eval(self):
        """Evaluations-Ergebnisse sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-nanoGPT", offline=True)
        if tracker.is_active:
            tracker.log_eval(iter_num=500, train_loss=1.8, val_loss=2.1)
        tracker.finish()

    def test_log_checkpoint(self):
        """Checkpoint-Info sollte ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-nanoGPT", offline=True)
        if tracker.is_active:
            tracker.log_checkpoint(iter_num=1000, val_loss=1.5, is_best=True)
        tracker.finish()

    def test_log_metrics(self):
        """Allgemeine Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-nanoGPT", offline=True)
        if tracker.is_active:
            tracker.log({"custom_metric": 0.95})
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden und doppeltes finish() sollte safe sein."""
        tracker = WandBTracker(project="test-nanoGPT", offline=True)
        tracker.finish()
        tracker.finish()
        assert not tracker.is_active

    def test_multiple_steps(self):
        """Mehrere Trainings-Schritte sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-nanoGPT", offline=True)
        if tracker.is_active:
            for i in range(10):
                tracker.log_step(iter_num=i * 100, train_loss=3.0 - i * 0.1,
                                val_loss=3.2 - i * 0.08, lr=1e-4 * (0.99 ** i))
        tracker.finish()

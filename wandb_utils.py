"""
W&B Experiment Tracking für nanoGPT
=====================================
Integriert Weights & Biases in das nanoGPT-Training.
Loggt Trainingsmetriken, Modell-Architektur und Hyperparameter.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="nanoGPT", config={...})
    tracker.log_step(iter_num=100, train_loss=2.5, val_loss=2.8, lr=1e-4, mfu=0.5)
    tracker.finish()
"""

import os
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für nanoGPT-Training.

    Features:
    - Trainings-Metriken (Loss, Val-Loss, LR, MFU)
    - Modell-Architektur-Logging
    - Hyperparameter-Tracking
    - Checkpoint-Logging
    """

    def __init__(
        self,
        project: str = "nanoGPT",
        config: dict | None = None,
        tags: list | None = None,
        group: str | None = None,
        job_type: str = "train",
        notes: str | None = None,
        offline: bool = False,
    ):
        self.project = project
        self.run = None
        self._start_time = time.time()

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["gpt", "transformer", "language-model"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:  # noqa: S110, BLE001
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int | None = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_step(
        self,
        iter_num: int,
        train_loss: float,
        val_loss: float | None = None,
        lr: float | None = None,
        mfu: float | None = None,
        dt_ms: float | None = None,
    ):
        """Loggt einen Trainings-Schritt."""
        metrics = {
            "train/loss": train_loss,
            "train/iter": iter_num,
        }
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if lr is not None:
            metrics["train/learning_rate"] = lr
        if mfu is not None:
            metrics["train/mfu"] = mfu
        if dt_ms is not None:
            metrics["train/step_time_ms"] = dt_ms
        self.log(metrics, step=iter_num)

    def log_eval(self, iter_num: int, train_loss: float, val_loss: float):
        """Loggt Evaluations-Ergebnisse."""
        self.log({
            "eval/train_loss": train_loss,
            "eval/val_loss": val_loss,
            "eval/iter": iter_num,
        }, step=iter_num)

    def log_checkpoint(self, iter_num: int, val_loss: float, is_best: bool = False):
        """Loggt Checkpoint-Info."""
        self.log({
            "checkpoint/iter": iter_num,
            "checkpoint/val_loss": val_loss,
            "checkpoint/is_best": 1 if is_best else 0,
        }, step=iter_num)

    def finish(self):
        """Beendet den W&B-Run."""
        elapsed = time.time() - self._start_time
        if self.run:
            self.log({"total_time_seconds": elapsed})
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None

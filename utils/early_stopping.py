#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track validation loss and stop after repeated non-improvement."""

    patience: int = 7
    min_delta: float = 0.0
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float, epoch: int | None = None) -> bool:
        if self.is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.should_stop = False
        else:
            self.num_bad_epochs += 1
            self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.patience = int(state_dict.get("patience", self.patience))
        self.min_delta = float(state_dict.get("min_delta", self.min_delta))
        self.mode = state_dict.get("mode", self.mode)
        self.best_score = state_dict.get("best_score", self.best_score)
        self.best_epoch = state_dict.get("best_epoch", self.best_epoch)
        self.num_bad_epochs = int(state_dict.get("num_bad_epochs", self.num_bad_epochs))
        self.should_stop = bool(state_dict.get("should_stop", self.should_stop))

from typing import Any
import sys

from pytorch_lightning.callbacks.progress.tqdm_progress import (
    Tqdm,
    TQDMProgressBar,
)
from pytorch_lightning import Trainer, LightningModule


class LogBar(TQDMProgressBar):
    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.main_progress_bar.total = pl_module.episode_length
        self.main_progress_bar.set_description(
            f"Episode {pl_module.episode_index}"
        )

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *_: Any
    ) -> None:
        if self._should_update(self.train_batch_idx):
            self.main_progress_bar.n = pl_module.step_index + 1
            self.main_progress_bar.set_postfix(
                self.get_metrics(trainer, pl_module)
            )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def on_test_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.main_progress_bar.total = pl_module.episode_length
        self.main_progress_bar.set_description(
            f"Episode {pl_module.episode_index}"
        )

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *_: Any
    ) -> None:
        if self._should_update(self.train_batch_idx):
            self.main_progress_bar.n = pl_module.step_index + 1
            self.main_progress_bar.set_postfix(
                self.get_metrics(trainer, pl_module)
            )

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Infering",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

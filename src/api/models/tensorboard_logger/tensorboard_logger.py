from __future__ import annotations

import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_path: pathlib.Path):
        self.log_path = log_path

    def log_metrics(
            self,
            scores: dict[str, float],
            epoch: int,
            stage: str
    ) -> None:
        with SummaryWriter(log_dir=self.log_path.as_posix()) as writer:
            for name, val in scores.items():
                writer.add_scalar(f'{name}/{stage}', val, epoch)

    def log_images(
            self,
            img: torch.Tensor,
            gt: torch.Tensor,
            pred: torch.Tensor,
            id_model: str,
            epoch: int,
            stage: str
    ):
        with SummaryWriter(log_dir=self.log_path.as_posix()) as writer:
            writer.add_image(f'{id_model}_{stage}/Image', img, epoch)
            writer.add_image(f'{id_model}_{stage}/Ground Truth', gt, epoch)
            writer.add_image(f'{id_model}_{stage}/Predict', pred, epoch)

    def log_hps(
            self,
            hparams: dict[str, str | float | int],
            scores: dict[str, float]
    ):
        with SummaryWriter(log_dir=self.log_path.as_posix()) as writer:
            hparam_scores = {f'hparam/{score}': scores[score] for score in scores}
            writer.add_hparams(hparams, hparam_scores, run_name='./')
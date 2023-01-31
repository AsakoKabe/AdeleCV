from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import fbeta_score, f1_score, iou_score, accuracy, recall, precision
from torch.utils.tensorboard import SummaryWriter

from models.base.BaseModel import BaseModel
from models.semantic.utils import get_preprocessing


class SegmentationModel(BaseModel):
    num_model = 0

    def __init__(
            self,
            model,
            optimizer,
            lr: float,
            loss_fn,
            num_classes,
            device,
            img_size,
            encoder_name='timm-mobilenetv3_small_minimal_100',
            encoder_weights='imagenet',
    ):
        super().__init__()
        SegmentationModel.num_model += 1
        self.device = device
        self._torch_model = model(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        self.num_classes = num_classes
        self._torch_model.to(self.device)
        self._transforms = get_preprocessing(
            get_preprocessing_fn(encoder_name, pretrained=encoder_weights), img_size
        )
        self.optimizer = optimizer(self._torch_model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.lr = lr
        self.count_epoch = 0
        self.writer = SummaryWriter(log_dir=f'../logs/{self.__str__()}')
        self.metrics = [
            fbeta_score, f1_score,
            iou_score, accuracy,
            precision, recall,
        ]

    def train_step(self, train_ds):
        self.train_mode()
        self.count_epoch += 1
        scores = defaultdict(float)
        for x_batch, y_batch in train_ds:
            self.optimizer.zero_grad()
            pred = self._torch_model(x_batch.to(self.device))
            loss = self.loss_fn(pred, y_batch.long().to(self.device))
            loss.backward()
            self.optimizer.step()
            scores['loss'] += loss.detach().cpu().numpy() / len(train_ds)
            for score, val in self._compute_metrics(y_batch, pred).items():
                scores[score] += val / len(train_ds)
        self._log_metrics(scores, self.count_epoch, 'Train')

        return scores['loss']

    def val_step(self, val_ds):
        self.eval_mode()
        scores = defaultdict(float)
        for x_batch, y_batch in val_ds:
            with torch.no_grad():
                pred = self._torch_model(x_batch.to(self.device))
                loss = self.loss_fn(pred, y_batch.long().to(self.device))
                scores['loss'] += loss.detach().cpu().numpy() / len(val_ds)
                for score, val in self._compute_metrics(y_batch, pred).items():
                    scores[score] += val / len(val_ds)

        self._log_metrics(scores, self.count_epoch, 'Valid')

        return scores['loss']

    def log_test_metrics(self, test_ds):
        self.eval_mode()
        scores = defaultdict(float)
        for x_batch, y_batch in test_ds:
            with torch.no_grad():
                pred = self._torch_model(x_batch.to(self.device))
                for score, val in self._compute_metrics(y_batch, pred).items():
                    scores[score] += val / len(test_ds)

        self._log_metrics(scores, 1, 'Test')

    def predict(self, img):
        self.eval_mode()
        img = self.transforms(image=img)['image']
        img = torch.Tensor(img).float().unsqueeze(0)

        with torch.no_grad():
            pred = self._torch_model(img.to(self.device))

        return pred

    def _log_metrics(self, scores, epoch, stage):
        for name, val in scores.items():
            self.writer.add_scalar(f'{stage}/{name}', val, epoch)

    def _compute_metrics(self, y_true, y_pred):
        tp, fp, fn, tn = smp.metrics.get_stats(
            y_pred.to(self.device),
            y_true.to(self.device).long(),
            mode='binary',
            threshold=0.5
        )
        scores = {}
        for metric in self.metrics:
            scores[metric.__name__] = metric(tp, fp, fn, tn, reduction='macro-imagewise')

        return scores

    def __str__(self):
        return f'{self.num_model}_{self._torch_model.__class__.__name__}_{self.optimizer.__class__.__name__}_' \
               f'{self.loss_fn.__class__.__name__}_lr={str(self.lr).replace(".", ",")}'

    def train_mode(self):
        self._torch_model.train()

    def eval_mode(self):
        self._torch_model.eval()

    @property
    def transforms(self):
        return self._transforms

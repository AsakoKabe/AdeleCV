from collections import defaultdict

import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import fbeta_score, f1_score, iou_score, accuracy, recall, precision

from api.models.base import BaseModel
from api.logs import get_logger
from .utils import get_preprocessing, denormalize


class SegmentationModel(BaseModel):
    def __init__(
            self,
            model,
            encoder_name,
            pretrained_weight,
            optimizer,
            lr: float,
            loss_fn,
            num_classes,
            num_epoch,
            device,
            img_size,
    ):
        super().__init__(
            model=model(
                encoder_name=encoder_name,
                encoder_weights=pretrained_weight,
                in_channels=3,
                classes=num_classes,
            ),
            optimizer=optimizer,
            lr=lr,
            loss_fn=loss_fn,
            transforms=get_preprocessing(
                get_preprocessing_fn(encoder_name, pretrained=pretrained_weight) if pretrained_weight else None,
                img_size
            ),
            metrics=[
                fbeta_score, f1_score,
                iou_score, accuracy,
                precision, recall,
            ],
            num_classes=num_classes,
            num_epoch=num_epoch,
            device=device,
        )
        self._encoder_name = encoder_name
        self._pretrained_weight = pretrained_weight

    def train_step(self, train_ds):
        self.train_mode()
        self._curr_epoch += 1
        scores = defaultdict(float)
        for x_batch, y_batch in train_ds:
            self._optimizer.zero_grad()
            pred = self._torch_model(x_batch.to(self.device))
            loss = self._loss_fn(pred, y_batch.long().to(self.device))
            loss.backward()
            self._optimizer.step()
            scores['loss'] += loss.detach().cpu().numpy() / len(train_ds)
            for score, val in self._compute_metrics(y_batch, pred).items():
                scores[score] += val / len(train_ds)

        self._logger.log_metrics(scores, self._curr_epoch, 'Train')
        self._logger.log_images(
            *self._get_images_for_logging(train_ds),
            id_model=self._id,
            epoch=self._curr_epoch,
            stage='Train'
        )

        return scores

    def _get_images_for_logging(self, dataset, index=0):
        img = torch.Tensor(dataset.dataset[index][0]).to(self.device)
        gt = torch.Tensor(dataset.dataset[index][1]).to(self.device)
        pred = self._torch_model(img.unsqueeze(0))

        img_denormalized = denormalize(img)
        gt_merged = torch.argmax(gt, dim=0).unsqueeze(0)
        pred_merged = torch.argmax(pred[0], dim=0).unsqueeze(0)

        return img_denormalized, gt_merged, pred_merged

    def _val(self, val_ds):
        self.eval_mode()
        scores = defaultdict(float)
        for x_batch, y_batch in val_ds:
            with torch.no_grad():
                pred = self._torch_model(x_batch.to(self.device))
                loss = self._loss_fn(pred, y_batch.long().to(self.device))
                scores['loss'] += loss.detach().cpu().numpy() / len(val_ds)
                for score, val in self._compute_metrics(y_batch, pred).items():
                    scores[score] += val / len(val_ds)

        return scores

    def val_step(self, val_ds):
        scores = self._val(val_ds)

        self._logger.log_metrics(scores, self._curr_epoch, 'Valid')
        self._logger.log_images(
            *self._get_images_for_logging(val_ds),
            id_model=self._id,
            epoch=self._curr_epoch,
            stage='Valid'
        )

        return scores

    def log_test(self, test_ds):
        scores = self._val(test_ds)
        hparams = self._get_hparams()

        self._logger.log_metrics(scores, 1, 'Test')
        self._logger.log_hps(hparams, scores)
        self._logger.log_images(
            *self._get_images_for_logging(test_ds),
            id_model=self._id,
            epoch=self._curr_epoch,
            stage='Test'
        )
        self._save_stats_model(hparams, scores)
        logger = get_logger()
        logger.info(f"Model {self.__str__()} trained with test loss {scores['loss']}")

    def predict(self, img):
        self.eval_mode()
        img = self.transforms(image=img)['image']
        img = torch.Tensor(img).float().unsqueeze(0)

        with torch.no_grad():
            pred = self._torch_model(img.to(self.device))

        return pred

    def _compute_metrics(self, y_true, y_pred):
        tp, fp, fn, tn = smp.metrics.get_stats(
            y_pred.to(self.device),
            y_true.to(self.device).long(),
            mode='binary',
            threshold=0.5
        )
        scores = {}
        for metric in self._metrics:
            scores[metric.__name__] = metric(tp, fp, fn, tn, reduction='macro-imagewise')

        return scores

    def __str__(self):
        return f'{self._id}_{self._torch_model.__class__.__name__}_{self._encoder_name}_{self._pretrained_weight}_' \
               f'{self._optimizer.__class__.__name__}_' \
               f'{self._loss_fn.__class__.__name__}_lr={str(self._lr).replace(".", ",")}'

    def _get_hparams(self):
        return {
            'architecture': self._torch_model.__class__.__name__,
            'encoder': self._encoder_name,
            'pretrained_weight': self._pretrained_weight,
            'lr': self._lr,
            'optimizer': self._optimizer.__class__.__name__,
            'loss_fn': self._loss_fn.__class__.__name__,
            'num_epoch': self._num_epoch,
        }

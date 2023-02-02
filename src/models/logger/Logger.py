from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log_metrics(self, scores, epoch, stage):
        with SummaryWriter(log_dir=self.log_path) as w:
            for name, val in scores.items():
                w.add_scalar(f'{name}/{stage}', val, epoch)

    def log_images(self, img, gt, pred, id_model, epoch, stage):
        with SummaryWriter(log_dir=self.log_path) as w:
            w.add_image(f'{id_model}_{stage}/Image', img, epoch)
            w.add_image(f'{id_model}_{stage}/Ground Truth', gt, epoch)
            w.add_image(f'{id_model}_{stage}/Predict', pred, epoch)

    def log_hps(self, hparams, scores):
        with SummaryWriter(log_dir=self.log_path) as w:
            hparam_scores = {f'hparam/{score}': scores[score] for score in scores}
            w.add_hparams(hparams, hparam_scores, run_name='./')

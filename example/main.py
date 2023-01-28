from data.dataset.SegmentationDataset import \
    SegmentationDataset
from data.dataset.types import ImageMaskSemantic
from task.SegmentationTask import SegmentationTask
from train.trainer import Trainer

if __name__ == '__main__':
    # dataset = SemanticDataset(r'F:\dataset\coco', COCOSemantic, (640, 640))
    dataset = SegmentationDataset(r'F:\dataset\ph2', ImageMaskSemantic, (256, 256))
    task = SegmentationTask(dataset)
    trainer = Trainer(task)

    trainer.run()


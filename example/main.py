from api.data.segmentations.dataset import \
    SegmentationDataset
from api.data.segmentations.types import ImageMask
from api.task.SegmentationTask import SegmentationTask
from train.trainer import Trainer

if __name__ == '__main__':
    # segmentations = SemanticDataset(r'F:\segmentations\coco', COCOSemantic, (640, 640))
    dataset = SegmentationDataset(r'F:\dataset\ph2', ImageMask, (256, 256))
    task = SegmentationTask(dataset)
    trainer = Trainer(task)

    trainer.run()


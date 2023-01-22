from data.dataset.SemanticSegmentationDataset import \
    SemanticDataset
from data.dataset.types import ImageMaskSemantic
from task.SemanticSegmentationTask import SemanticSegmentationTask
from train.trainer import Trainer

if __name__ == '__main__':
    # dataset = SemanticDataset(r'F:\dataset\coco', COCOSemantic, (640, 640))
    dataset = SemanticDataset(r'F:\dataset\ph2', ImageMaskSemantic, (256, 256))
    task = SemanticSegmentationTask(dataset)
    trainer = Trainer(task)

    trainer.run()


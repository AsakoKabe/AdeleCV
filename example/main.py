from data.dataset.SemanticSegmentationDataset import \
    SemanticSegmentationDataset
from train.task.SemanticSegmentationTask import SemanticSegmentationTask
from train.trainer import Trainer

if __name__ == '__main__':
    dataset = SemanticSegmentationDataset(r'F:\dataset\ph2')
    task = SemanticSegmentationTask(dataset)
    trainer = Trainer(task)

    trainer.run()


from train.task.base import BaseTask


class SemanticSegmentationTask(BaseTask):
    def __init__(self, dataset):
        super(SemanticSegmentationTask, self).__init__(
            dataset=dataset
        )


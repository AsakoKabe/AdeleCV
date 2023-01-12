from data.dataset.SementicTorchDataset import SemanticTorchDataset
from data.dataset.types import DatasetType


class SemanticDataset:
    def __init__(
            self,
            dataset_dir,
            dataset_type,
            img_size,  # height, width
            split=(0.7, 0.2, 0.1),
            batch_size=5
    ):
        self.dataset_dir = dataset_dir
        self.dataset = dataset_type.create_dataset(self.dataset_dir)
        self.split = split
        self.train = None
        self.val = None
        self.test = None
        self.batch_size = batch_size
        self.dataset.info['img_size'] = img_size
        self.num_classes = len(self.dataset.default_mask_targets)

        self._split_dataset()
        self._create_torch_datasets()

    def _split_dataset(self):
        self.dataset.take(
            int(self.split[0] * len(self.dataset))
        ).tag_samples("train")
        self.dataset.match_tags(
            "train",
            bool=False
        ).tag_samples("valid_test")
        self.dataset.match_tags("valid_test").take(
            int(self.split[1] * len(self.dataset))
        ).tag_samples("valid")
        self.dataset.match_tags(
            ["train", "valid"],
            bool=False
        ).tag_samples("test")
        self.dataset.untag_samples('valid_test')

    def _create_torch_datasets(self):
        self.train = SemanticTorchDataset(self.dataset.match_tags('train'))
        self.val = SemanticTorchDataset(self.dataset.match_tags('valid'))
        self.test = SemanticTorchDataset(self.dataset.match_tags('test'))

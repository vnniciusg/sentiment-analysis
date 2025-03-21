from torch.utils.data import Dataset


class TwitterSentimentDataset(Dataset):

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item["cleaned_text"], item["label"]

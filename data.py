import pandas as pd, os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ActionPersonDataset(Dataset):
    def __init__(self, data_root, labels_csv, img_size=224, is_train=True):
        self.df = pd.read_csv(labels_csv)
        self.root = data_root
        aug = [T.Resize((img_size, img_size)), T.ToTensor(),
               T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        if is_train:
            aug.insert(0, T.RandomHorizontalFlip(p=0.5))
        self.tf = T.Compose(aug)
        self.actions = sorted(self.df["action_label"].unique())
        self.a2i = {a:i for i,a in enumerate(self.actions)}

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, r["filepath"])).convert("RGB")
        x = self.tf(img)
        y_action = self.a2i[r["action_label"]]
        y_person = float(r["person_present"])
        return x, y_action, y_person

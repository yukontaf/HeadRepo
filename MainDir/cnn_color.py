
# %% codecell
import torch.utils.model_zoo as model_zoo
from PIL import Image
import os
from pprint import pprint
import matplotlib.pyplot as plt
import snoop
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pretrainedmodels import utils
import pretrainedmodels
from tqdm import tqdm
import gc
import torch
import torch.nn as nn

device = torch.device("cpu:0")

X = torch.randn(5, 100, device=device)

model = nn.Sequential(nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 8))

model = model.to(device)

model(X)


# %% codecell


def monitor_gpu():
    if device.type == "cuda":
        print("Memory Usage:")
        print("Allocated:", round(
            torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(
            torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
    else:
        print("No GPU found.")


# %%
inception = pretrainedmodels.__dict__["inceptionresnetv2"](
    num_classes=1001,
    pretrained="imagenet+background"
)

# %% codecell

load_img = utils.LoadImage()
tf_img = utils.TransformImage(inception)


class ColorizationDataset(Dataset):
    def __init__(self, path, transform_x, transform_y, transform_inc):
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.transform_inc = transform_inc

        self.filenames = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".JPG"):
                    self.filenames.append(os.path.join(root, file))

        self.images = []
        for filename in tqdm(self.filenames):
            try:
                with Image.open(filename) as image:
                    self.images.append(image.copy())
            except:
                pass
                # print('Could not load image:', filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        Y = self.transform_y(img)
        X = self.transform_x(Y)
        e = tf_img(self.transform_inc(img))
        return X, Y, e


# Чтобы подавать картинки на вход нейросети, нужно их перевести в тензоры, причём одинакового размера.
# %% codecell
transform_all = transforms.Compose(
    [
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


def to_grayscale(x):
    return 1 - (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114).view(1, 256, 256)


transform_inc = transforms.Compose([transforms.CenterCrop(299)])


# %% codecell
inception = inception.to(device)
inception.eval()


# %% codecell
dataset = ColorizationDataset("/Users/glebsokolov/MainDir/universum-photos", to_grayscale, transform_all, transform_inc
                              )
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# %% codecell


class Colorizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.MaxPool2d((2, 2)),  # squeeze
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.MaxPool2d((2, 2)),  # squeeze
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),  # squeeze
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1258, 256, (3, 3), padding=1),
            nn.LeakyReLU(),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, (3, 3), padding=1),
            nn.Tanh(),
        )

    #     @snoop
    def forward(self, x, e):
        h = self.encoder(x)
        h = torch.cat((h, squeeze_input(x)), 1)
        with torch.no_grad():
            embedding = inception(e)

        embedding = embedding.view(-1, 1001, 1, 1)

        rows = torch.cat([embedding] * 64, dim=3)
        embedding_block = torch.cat([rows] * 64, dim=2)
        fusion_block = torch.cat([h, embedding_block], dim=1)
        h = self.decoder(fusion_block)
        return h


# %% codecell
def squeeze_input(x):
    squeezer = nn.Sequential(nn.MaxPool2d(
        2, 2),  nn.MaxPool2d(2, 2))
    return squeezer(x)


# %% codecell
gc.collect()
# %% codecell
num_epochs = 100
lr = 1e-3

model = Colorizer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()  # тут можно поиграться с лоссами
# %% codecell
torch.cuda.empty_cache()
monitor_gpu()

# %%
model.load_state_dict(torch.load(
    '/Users/glebsokolov/Downloads/chkpt-2.zip', map_location=torch.device('cpu')))

# %% codecell
num_epochs = 5
history = []
for epoch in tqdm(range(num_epochs)):
    for x, y, e in loader:
        x, y, e = x.to(device), y.to(device), e.to(device)
        optimizer.zero_grad()
        output = model(x, e)
        loss = criterion(output, y)
        history.append(loss.item())
        loss.backward()
        optimizer.step()

# %% codecell


def to_numpy_image(img):
    return img.detach().cpu().view(3, 256, 256).transpose(0, 1).transpose(1, 2).numpy()


# %% codecell
for t in range(10):
    img_gray, img_true, e = dataset[t]
    img_pred = model(
        img_gray.to(device).view(1, 1, 256, 256), e.to(
            device).view(1, 3, 299, 299)
    )
    img_pred = to_numpy_image(img_pred)
    # теперь это numpy-евский ndarray размера (128, 128, 3)
    plt.figure(figsize=(10, 10))

    plt.subplot(141)
    plt.axis("off")
    plt.set_cmap("Greys")
    plt.imshow(img_gray.reshape((256, 256)))

    plt.subplot(142)
    plt.axis("off")
    plt.imshow(img_pred.reshape((256, 256, 3)))

    plt.subplot(143)
    plt.axis("off")
    plt.imshow(to_numpy_image(img_true))

    plt.show()

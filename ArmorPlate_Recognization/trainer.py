import torch
from torch import nn
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from time import clock

class CustomImgDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """
    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags'].astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(np.array(self.y_train[index])).long()
        return img, label

    def __len__(self):
        return len(self.X_train.index)

                 ###############################
################################SET################################
                 ###############################

# NETWORK
# NETWORK
class CNN(nn.Module):
    # for 55 x 125
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2),   # 55 x 125 x 1 -> (55-3)/2+1=32 x (125-3)/2+1=62 x 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # 16x31
        )
        self.ln = nn.Sequential(
            nn.Linear(806, 16),
            nn.Softmax(1),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.ln(out)
        return out


if __name__ == "__main__":
    # DEVICE
        # ########## !!! LOOK HERE !!! ############ #
    use_gpu = 0
        # ######################################### #
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # IMG INFO
    IMG_PATH = 'AP/'
    IMG_EXT = '.png'
    DATA_INFO = 'AP_info.csv'

    # PARAMETERS
    num_epoachs = 10
    batch_size = 5
    times4print = 1000 / batch_size  # time for print (I print the info for every * batches)
    classes = ['0', '1']
    num_classes = 2
    learning_rate = 0.01

    dataset = CustomImgDataset(DATA_INFO, IMG_PATH, IMG_EXT, transforms.ToTensor())

    # LOADER
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=2,
                                               num_workers=2,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              num_workers=2,
                                              shuffle=True)

    # MODEL, LOSS FUNC AND OPTIMISER
    model = CNN(num_classes)
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # TRAIN
    total_steps = len(train_loader)
    for epoach in range(num_epoachs):
        loss_accumulation = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss = loss_func(out, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_accumulation += loss.item()
            if (i + 1) % times4print == 0:
                print(f"[{epoach+1}/{num_epoachs}]: -> [{i+1}/{total_steps}] -> loss: {loss_accumulation/times4print}")
                loss_accumulation = 0

    # TEST
    model.eval()

    time_total = 0.

    with torch.no_grad():
        class_correct = list(0. for i in range(num_classes))
        class_total = class_correct.copy()
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            a = clock()
            out = model(imgs)

            time_total += clock()-a

            _, predicted = torch.max(out, 1)
            ans_batch = (predicted == labels).squeeze()
            for k, label in enumerate(labels):
                if ans_batch.item() == 1: # right
                    class_correct[label] += 1
                class_total[label] += 1
        print(class_total)
        if sum(class_total) != 0:
            print(f"\tFINAL ACCURACY: {100 * sum(class_correct)/sum(class_total)}% -> {class_correct}/{class_total}")
        for i in range(num_classes):
            if class_total[i] != 0:
                print(f"\t[{classes[i]}] : {100 * class_correct[i]/class_total[i]}% -> {class_correct[i]}/{class_total[i]}")

    print(f"\tFor prediction, each image runs for {time_total/sum(class_total)}   s")

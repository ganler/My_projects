# NN experiment on AP_dataset

> Given by Jaway Liu. 
>
> 25th Nov.

## Comments

> The num of the data is rich.
>
> **size**: 55 $\times$ 125 $\times$ 1
>
> **format**: png
>
> **num of 0 samples**: 763
>
> **num of 1 samples**: 351

#### Problems

- Folder 0 & folder 1 have same pics(in the same name).
- No label in `.csv` or other format which is convinient for data loading.

> For this problem I think we can use some python code to rename them in `1-xxx` format, and gather them together in one directory. I'll show the code later.

- Acctually the nums of 0 samples and 1 samples should be the same.(You can refer to Andrew Ng's book for more detail)

> I just ignore this item as I want test my model as soon as possible and see what happened if the data is not balanced(Sometimes unbalanced data will make the model more smart, see more details in the following introduction).

## What did I do on this data

### Python module required

- NumPy
- os
- PyTorch
- TorchVision
- Pandas
- Shutil
- PIL

### My workspace

It will look like this after running the codes:

![img](https://s1.ax1x.com/2018/11/26/FAAdFP.png)

`0` & `1` are original datasets and its category is the name of the directory itself. I used the `foo.py` to convert the name of the files in `0` & `1`, cause some files in different directory share the same name which will cause some bugs when I load the dataset after my combining the `0` and `1` together into `AP`.

Here is the `foo.py`:

```python
import os
import pandas as pd
import shutil

paths = ['0', '1']
dict_ = {'image_name': [], 'tags': []}
new_dir = 'AP'

os.mkdir(new_dir)

for tag, path in enumerate(paths):
    for i, file in enumerate(os.listdir(path)):
        os.rename(os.path.join(path, file), os.path.join(path, path + '-' + str(i) + ".png"))
        dict_['tags'].append(str(tag))
        dict_['image_name'].append(path + '-' + str(i))
        shutil.copy(os.path.join(path, path + '-' + str(i) + ".png"), new_dir)

pd.DataFrame(dict_).to_csv('AP_info.csv')
```

### Prediction

#### Input

**Gray Image**: 55(row) by 125(column) by 1(channel).

#### output

A 2-d array markinig the scores for each classes(We only have 2 classes which are 0 and 1, and `0` stands for positive class).

#### NN frame

> This part is not that necessary as it can be fine-tuned till we get a satisfactory model.

I tried a lot of tricks for many hours and I got some good ideas:

- Use more filters, not just 1 filter for every Conv2d.
- Use BN.
- Use more linear layers, and use SoftMax between them.
- More data, better model.

All this I've done can help the model be more robustic on different classifications with higher speed. 

Less loss is not a good thing cause it may just show that you're on the overfitting way. All we care is its performance on test data. Like this:

```python
FINAL ACCURACY: 95.32374100719424% -> [727.0, 333.0]/[762.0, 350.0]
[0] : 95.40682414698163% -> 727.0/762.0
[1] : 95.14285714285714% -> 333.0/350.0
```

**Advantages**

- Fast(both in training part and prediction part)
- Robustic(Both good on `0` and `1`)
- 

**My frame**

> See what the parameters mean in PyTorch's official doc.

```python
nn.Conv2d(1, 2, kernel_size=3, stride=2)   # 55 x 125 x 1 -> (55-3)/2+1=32 x (125-3)/2+1=62 x 1
nn.BatchNorm2d(2)
nn.ReLU()
nn.MaxPool2d(2, 2) # 16x31

nn.Conv2d(2, 2, kernel_size=(2, 3))  # 16 x 31 x 1 -> (16-2)/2+1 x (31-3)/2+1 x 1 = 8 x 15 x 1
nn.BatchNorm2d(2)
nn.ReLU()
nn.MaxPool2d(2, 3) # 4 x 5 x 1

nn.Linear(40*2, 8)
nn.Softmax()
nn.Linear(8, num_classes) # num_class is 2
```

> **Other detials I want to share:**
>
> - This problem is quite simple, so we don't need complex layers.
> - If you want use MSE loss you need to tune the code.
> - It seems to perform better on test dataset when the loss is around 0.15.
> - I also tried some "strict" models on it which can make the loss to be under 0.01, but it's actually nothing but overfitting.
> - Epoachs around 10 will be enough.
> - About unbalanced data. At least in this turn, I found that more data will help optimize the parameter as long as the model is good.(I also tried some bad models, and unbalanced data will make the model stupid. The stupid model thought that all images are in `1` class.)

**Code**

```python
import torch
from torch import nn
import torchvision
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np

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

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMG INFO
IMG_PATH = 'AP/'
IMG_EXT = '.png'
DATA_INFO = 'AP_info.csv'

# PARAMETERS
num_epoachs = 10
batch_size = 5
times4print = 1000/batch_size # time for print (I print the info for every * batches)
classes = ['0', '1']
num_classes = 2
learning_rate = 0.001

dataset = CustomImgDataset(DATA_INFO, IMG_PATH, IMG_EXT, transforms.ToTensor())


# LOADER
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           num_workers=2,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          num_workers=2,
                                          shuffle=True)

# NETWORK
class CNN(nn.Module):
    # for 55 x 125
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2),   # 55 x 125 x 1 -> (55-3)/2+1=32 x (125-3)/2+1=62 x 1
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 16x31
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(2, 3)),  # 16 x 31 x 1 -> (16-2)/2+1 x (31-3)/2+1 x 1 = 8 x 15 x 1
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2, 3) # 4 x 5 x 1
        )
        self.ln = nn.Sequential(
            nn.Linear(40*2, 8),
            nn.Softmax(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.ln(out)
        return out

# MODEL, LOSS FUNC AND OPTIMISER
model = CNN(num_classes)
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adamax(model.parameters(), lr=learning_rate)

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
with torch.no_grad():
    class_correct = list(0. for i in range(num_classes))
    class_total = class_correct.copy()
    for imgs, labels in test_loader:
        out = model(imgs)
        _, predicted = torch.max(out, 1)
        ans_batch = (predicted == labels).squeeze()
        for k, label in enumerate(labels):
            if ans_batch[k][0] == 1: # right
                class_correct[label] += 1
            class_total[label] += 1
    print(class_total)
    if sum(class_total) != 0:
        print(f"FINAL ACCURACY: {100 * sum(class_correct)/sum(class_total)}% -> {class_correct}/{class_total}")
    for i in range(num_classes):
        if class_total[i] != 0:
            print(f"[{classes[i]}] : {100 * class_correct[i]/class_total[i]}% -> {class_correct[i]}/{class_total[i]}")
```




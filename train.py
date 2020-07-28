import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
import numpy as np
from skimage import io, transform
import torch.nn as nn
import torch.optim as optim


class HandKeypointDataset(Dataset):
    """OpenPose Hand Landmarks dataset."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.landmark_paths = [i for i in os.listdir(self.root) if i.endswith(".json")]
        self.img_paths = [i[0:-5] + ".jpg" for i in self.landmark_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root, self.img_paths[idx])
        image = io.imread(img_path)

        landmark_path = os.path.join(self.root, self.landmark_paths[idx])
        landmark_data = json.load(open(landmark_path, 'r'))
        landmark_points = np.array(landmark_data['hand_pts'])[:, :2]
        landmarks = landmark_points.astype('float')
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class ModifiedNormalize(object):
    """Normalize ndarrays in sample, keep labels as same."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = transforms.functional.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).float()
        landmarks = torch.reshape(landmarks, (42,)).float()
        return {'image': image,
                'landmarks': landmarks}


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.pretrain_net = torchvision.models.squeezenet1_1(pretrained=True)
        self.base_net = self.pretrain_net.features
        self.pooling = nn.AvgPool2d(3)
        self.fc = nn.Linear(51200, num_classes)

    def forward(self, x):
        x = self.base_net(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    hand_keypoint_dataset = HandKeypointDataset(root="../datasets/KeypointDataset/",
                                                transform=transforms.Compose([Rescale((512, 512)),
                                                                              ToTensor(),
                                                                              ModifiedNormalize()
                                                                              ]))

    dataloader = DataLoader(hand_keypoint_dataset, batch_size=8, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    smodel = SqueezeNet(42).to(device)
    print(smodel)
    print(device)

    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(params=smodel.parameters(), lr=0.001, weight_decay=0.0005)

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    print('Started Training')
    for epoch in range(50):
        running_loss = 0.0

        # Run the training batches
        for b, train_f in enumerate(dataloader):
            y_train = train_f['landmarks'].to(device)
            x_train = train_f['image'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Apply the model
            y_pred = smodel(x_train)
            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Print interim results
            if b % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, b + 1, running_loss / 2000))
                running_loss = 0.0

        print('Epoch End Loss:', running_loss)

    out_path = "key_net.pth"
    torch.save(smodel.state_dict(), out_path)

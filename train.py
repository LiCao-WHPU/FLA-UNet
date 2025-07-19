from model.FLAUnet_model import FLA_UNet as UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os
import time
import cv2
from utils.utils_metrics import compute_mIoU, show_results



# Set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(f'USE GPU {'0'}')

#build the model
net = UNet(n_channels=1, n_classes=1).cuda()

#set the path
data_path = "C:/Users/zxdn/Desktop/data" 
test_dir="C:/Users/chenmingsong/Desktop/unetnnn/skin/Test_Images"
pred_dir="C:/Users/chenmingsong/Desktop/unetnnn/skin/results"
gt_dir="C:/Users/chenmingsong/Desktop/unetnnn/skin/Test_Labels"
miou_out_path = "results/"

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

#  initialization
batch_size=8
lr=0.00001
epoch = 200
MODE = 'Train'
# MODE = 'Test'

# We define the scheduler
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

# data loade
isbi_dataset = ISBI_Loader(data_path)
per_epoch_num = len(isbi_dataset) / batch_size
train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)


def train(net, train_loader, optimizer, epoch, criterion):
    with tqdm(total=epoch*per_epoch_num) as pbar:
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                pred = net(image)

                loss = criterion(pred, label)

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')
                loss.backward()
                optimizer.step()
                pbar.update(1)


def test(net, miou_mode, num_classes, name_classes, test_dir, pred_dir, gt_dir, miou_out_path):
    if miou_mode == 0 or miou_mode == 1:
        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]
        print("Get predict result.")

        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (512, 512))
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.cuda()
            pred = net(img_tensor)
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)



if MODE == 'Train': # 根据MODE变量的值选择训练或测试模式
    print("Start train...") 
    time_start=time.time()
    net.train()
    for epoch in range(1, epoch + 1):
        # train
       train(net, train_loader, optimizer, epoch, criterion)

    # train(net, optimizer, 50, scheduler) # 调用train函数开始训练过程 
    time_end=time.time() # 记录训练的总时间
    print('Total Time Cost: ',time_end-time_start)

elif MODE == 'Test':
        net.load_state_dict(torch.load('best_model_skin.pth'), strict=False) # seg, load net
        net.eval()
        miou_mode = 0
        num_classes = 2
        name_classes = ["background", "nidus"]
        test(net, miou_mode, num_classes, name_classes, test_dir, pred_dir, gt_dir, miou_out_path) # 进行测试，并获取平均交并比、预测结果和真实标签 
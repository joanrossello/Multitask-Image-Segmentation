"""
Author: Group work
Date: 2021/12/26
Description: baseline model U-net task 
"""

# import packages
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim
from models.unet_base import UNet
from loader import *
import math
import pandas as pd
from torchvision.utils import save_image
from utils import save_checkpoint,eval_metrics,plot_figure

def train_base(epoch_num, net, optimizer, batch_size, loss_fun, device, trainset, validset=None, save_cp=False):
    
    # initialize 
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc':[], 'val_iou':[]}
    img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    num_train_batch = math.ceil(len(trainset)/batch_size)
    if validset:
        num_val_batch = math.ceil(len(validset)/batch_size)
    
    
    print('Beginning training...')
    start_time = time.time()
    for epoch in range(epoch_num):
        
        epoch_time = time.time()
        train_loader = PetDatasetGenerator(trainset, batch_size, shuffle=True)
        train_loss = 0.0
        for i,data in enumerate(train_loader):
            # get data
            images, masks, labels, bboxes = data
            images, masks= images.to(device), masks.to(device)
            images = img_transform(images)
            
            # get output image through network
            out_image=net(images)
            
            # loss function
            loss=loss_fun(out_image,masks)
            train_loss += loss.item()
            
            # zero grad + backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training loss
            if i%20==0:
                print(f'{epoch}-{i}-train_loss===>>{loss.item()}')

                # visualize training images
                # _image=image[0]
                # _segment_image=masks[0]
                # _out_image=out_image[0]

                # img=torch.stack([_segment_image,_out_image],dim=0)
                # save_image(img,f'{save_path}/epoch_{epoch}/{i}.png')  
        # validation
        if validset:
            # initialize metrics
            val_loader = PetDatasetGenerator(validset, batch_size, shuffle=True)
            total_correct = 0
            total_pixels = 0
            iou_score = []
            valid_loss = 0.0
            with torch.no_grad():
                for i,data in enumerate(val_loader):
                    # load validation data
                    images, masks, labels, bboxes = data
                    images, masks = images.to(device), masks.to(device)
                    images = img_transform(images)
            
                    # get output image through network
                    preds=net(images)
                    
                    # loss function
                    loss=loss_fun(preds,masks)
                    valid_loss += loss.item()
                    
                    # evaluation metrics
                    preds = (preds > 0.5).float()
                    num_correct, num_pixels, iou = eval_metrics(preds, masks)
                    total_correct += num_correct
                    total_pixels += num_pixels
                    iou_score.append(iou)
            #calculate accuracy and iou 
            val_acc = total_correct/total_pixels
            val_iou = torch.cat(iou_score, dim=0).mean()
            print(f"Got {total_correct}/{total_pixels} with acc on epoch_{epoch} validation set {val_acc*100:.2f}")
            print(f'Got IoU score on on epoch_{epoch} validation set: {val_iou:.3f}')
                    
        # store the loss and acc to dict
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss/num_train_batch)
        if validset:
            history['val_loss'].append(valid_loss/num_val_batch)
            history['val_acc'].append(val_acc.item())
            history['val_iou'].append(val_iou.item())
        
        if save_cp:
            # save checkpoint model
            save_checkpoint(net,optimizer,'unet_base_epoch_{}.pt'.format(epoch))
            print('Model_epoch_{} saved'.format(epoch))
        # print time spent for each epoch
        epoch_time_spent = time.time() - epoch_time
        print('epoch {} time spent: {:.0f}m {:.0f}'.format(epoch,epoch_time_spent // 60 , epoch_time_spent % 60))
        
    torch.save(net.state_dict(), 'unet_base_final.pt')
    print('Model saved')
    # print time spent
    time_spent = time.time() - start_time
    print('Trainning complete in {:.0f}m {:.0f}s\n'.format(time_spent // 60 , time_spent % 60))

    return history


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # initialize
    epochs = 10
    lr = 1e-4
    batch_size = 5
    # get dataset
    trainset = PetDataset('datasets-oxpet/train/')
    validset = PetDataset('datasets-oxpet/val/')
    # define net model,optimizer and loss function
    net=UNet().double().to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    loss_fun=nn.BCEWithLogitsLoss()

    # training model 
    train_result = train_base(epochs, net, optimizer, batch_size, loss_fun, device, trainset, validset)
    
    # save training result to csv
    df = pd.DataFrame.from_dict(train_result,orient='index').T
    df.to_csv('baseline_res.csv',header=True)
    
    # plot training loss/acc
    plot_figure(train_result,'baseline_model_fig.png')
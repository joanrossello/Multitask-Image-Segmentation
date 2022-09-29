"""
Author: Group work
Date: 2021/1/4
Description: multi-task learning model training
"""

# import packages
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim
from models.unet_mtl import UNET_MTL
from loader import *
import math
import pandas as pd
from torchvision.utils import save_image
from utils import save_checkpoint,eval_metrics,plot_figure

# Function: multi-task learning training func
def train_mtl(epoch_num, net, optimizer, batch_size, loss_fun, weights, task_order, device, trainset, validset=None, save_cp=False, early_stop=False, name='mtl'):
    """main train function for multi-task learning task

    Args:
        epoch_num (int): number of epochs
        net (func): multi-task learning neural network
        optimizer (func): optimizer
        batch_size (int): the size of each mini-batch
        loss_fun (list): a list of loss functions. segementation task loss should always be the first element.
        weights (list): a list of scaling factors of loss functions. 
        task_order (list of int): a list of orders of ground truth tasks. 0: images, 1: masks, 2: labels, 3: bboxes. Ex. [1,2,3] is a 3 tasks experiment which ground truth for the first task is 'mask', ... 
        trainset (custom dataset): a custom training dataset
        validset (custom dataset, optional): a custom validation dataset. Defaults to None.
        save_cp (bool, optional): if true then save checkpoint. Defaults to False.
        early_stop(bool, optional): if true then using early stopping algorithm. Defaults to False
        name (str, optional):  name of the model. Defaults to 'mtl'.

    Returns:
        dict: a dictory containing training result
    """
    assert len(loss_fun) == len(weights) == len(task_order)
    # initialize 
    min_val_loss = 10000 # for early stopping stragety
    task_num = len(loss_fun)  # get number of tasks
    if task_num == 2:
        loss_func1, loss_func2 = loss_fun # notice segement loss will always be loss_func1
        w1, w2 = weights
    elif task_num == 3:
        loss_func1, loss_func2, loss_func3 = loss_fun
        w1, w2, w3 = weights
    else:
        print('task number for mtl model must be 2 or 3!')
        return None
    
    history = {'epoch': [], 'train_loss_total': [], 'train_loss_1': [],'train_loss_2': [],'train_loss_3': [],'val_loss': [], 'val_acc':[], 'val_iou':[]}
    img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    num_train_batch = math.ceil(len(trainset)/batch_size)
    if validset:
        num_val_batch = math.ceil(len(validset)/batch_size)
    
    
    print('Beginning training...')
    start_time = time.time()
    for epoch in range(epoch_num):
        
        epoch_time = time.time()
        train_loader = PetDatasetGenerator(trainset, batch_size, shuffle=True)
        train_loss, running_loss1, running_loss2 ,running_loss3 = 0,0,0,0
        for i,data in enumerate(train_loader):
            # get data
            images, masks, labels, bboxes = data
            images, masks, labels, bboxes = images.to(device), masks.to(device), labels.to(device), bboxes.to(device)
            images = img_transform(images)
            groud_truth = [images,masks,labels,bboxes]
            if task_num == 2:
                # get output image through network
                out_task1, out_task2 = net(images)
                
                # loss function
                task1_loss = loss_func1(out_task1, groud_truth[task_order[0]])
                running_loss1 += task1_loss.item()
                
                task2_loss = loss_func2(out_task2, groud_truth[task_order[1]])
                running_loss2 += task2_loss.item()
                
                # loss method
                loss = w1 * task1_loss + w2 * task2_loss
                train_loss += loss.item()
            else:
                # get output image through network
                out_task1, out_task2, out_task3 = net(images)
                
                # loss function
                task1_loss = loss_func1(out_task1, groud_truth[task_order[0]])
                running_loss1 += task1_loss.item()
                
                task2_loss = loss_func2(out_task2, groud_truth[task_order[1]])
                running_loss2 += task2_loss.item()
                
                task3_loss = loss_func3(out_task3, groud_truth[task_order[2]])
                running_loss3 += task3_loss.item()
                # loss method
                loss = w1*task1_loss + w2*task2_loss + w3*task3_loss
                train_loss += loss.item()
            
            # zero grad + backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print training loss
            if task_num == 2:
                if i%20==0:
                    print(f'{epoch}-{i}: TOTAL LOSS = {loss} | TASK1 LOSS = {task1_loss} | TASK2 LOSS = {task2_loss}')
            else:
                if i%20==0:
                    print(f'{epoch}-{i}: TOTAL LOSS = {loss} | TASK1 LOSS = {task1_loss} | TASK2 LOSS = {task2_loss} | TASK3 LOSS = {task3_loss}')
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
                    preds=net(images)[0]
                    
                    # loss function
                    loss=loss_func1(preds,masks)
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
        history['train_loss_total'].append(train_loss/num_train_batch)
        history['train_loss_1'].append(running_loss1/num_train_batch)
        history['train_loss_2'].append(running_loss2/num_train_batch)
        if task_num == 3:
            history['train_loss_3'].append(running_loss3/num_train_batch)
        if validset:
            history['val_loss'].append(valid_loss/num_val_batch)
            history['val_acc'].append(val_acc.item())
            history['val_iou'].append(val_iou.item())
        
        # early stopping 
        if early_stop: 
            cur_min_val_loss = history['val_loss'][-5:]
            if cur_min_val_loss > min_val_loss:
                break
            else:
                min_val_loss = cur_min_val_loss
        
        if save_cp:
            # save checkpoint model
            save_checkpoint(net,optimizer,'{}_epoch_{}.pt'.format(name,epoch))
            print('Model_epoch_{} saved'.format(epoch))
        # print time spent for each epoch
        epoch_time_spent = time.time() - epoch_time
        print('epoch {} time spent: {:.0f}m {:.0f}'.format(epoch,epoch_time_spent // 60 , epoch_time_spent % 60))
        
    torch.save(net.state_dict(), '{}_final.pt'.format(name))
    print('Model saved')
    # print time spent
    time_spent = time.time() - start_time
    print('Trainning complete in {:.0f}m {:.0f}s\n'.format(time_spent // 60 , time_spent % 60))

    return history


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # initialize
    epochs = 20
    lr = 1e-4
    batch_size = 5
    # get dataset
    trainset = PetDataset('datasets-oxpet/train/')
    validset = PetDataset('datasets-oxpet/val/')
    # define net model,optimizer and loss function
    net= UNET_MTL(in_ch=3, out_ch=1).double().to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    seg_criterion = torch.nn.BCEWithLogitsLoss()
    class_criterion = torch.nn.BCEWithLogitsLoss()
    bbox_criterion = torch.nn.MSELoss()
    loss_fun = [seg_criterion,class_criterion,bbox_criterion]
    
    # define the scaling factors and task order
    weights= [1,1,1] 
    task_order = [1,2,3] 
    
    # training model 
    train_result = train_mtl(epochs, net, optimizer, batch_size, loss_fun, weights, task_order, device, trainset, validset,name='mtl')
    
    # save training result to csv
    df = pd.DataFrame.from_dict(train_result,orient='index').T
    df.to_csv('mtl.csv',header=True)
    
    # plot training loss/acc
    # plot_figure(train_result,'mtl_model_fig.png')
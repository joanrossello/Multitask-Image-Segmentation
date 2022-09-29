"""
Author: Group work
Date: 2021/12/29
Description: utility functions for this project
"""

# import packages
from torch import optim
from torch.nn.modules import loss
import torchvision.transforms as transforms
import torch
from models.unet_mtl_rec import UNET_MTL
from loader import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Function: return the dice score
def dice_score(preds, masks):
    smooth = 1

    iflat = preds.view(-1)
    tflat = masks.view(-1)
    
    intersection = (iflat * tflat).sum()
    
    return (2. * intersection + smooth) /(preds.sum() + masks.sum() + smooth)
    
# Function: return the pixel accuracy and iou score
def eval_metrics(preds, masks):
    
    # calcualte pixel accuaray
    num_correct = (preds == masks).sum()
    num_pixels = torch.numel(preds)
    
    # calculate iou score
    preds_np = preds.squeeze(1).int()  # BATCH x 1 x H x W => BATCH x H x W
    masks_np = masks.squeeze(1).int()
    
    eps = 1e-6
    intersection = (preds_np & masks_np).float().sum((1, 2))
    union = (preds_np | masks_np).float().sum((1, 2))        
    
    iou = (intersection + eps) / (union + eps)  # use epsilon to avoid 0/0
    
    # calculate dice score
    # dice = 2* intersection / (preds_np.float().sum((1, 2)) + preds_np.float().sum((1, 2)))
    dice = dice_score(preds, masks)
    return num_correct, num_pixels, iou, dice
    

# Function: test accuracy/iou on model
def test(net, device, dataloader, mode='mtl'):
    save_path = 'test_image'
    img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # initialize metrics
    total_correct = 0
    total_pixels = 0
    iou_score = []
    dice_score = []
    
    # test func
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # get data
            images, masks, labels, bboxes = data
            images, masks, labels, bboxes = images.to(device), masks.to(device), labels.to(device), bboxes.to(device)
            images = img_transform(images)
            
            # get pred images
            if mode == 'mtl':
                preds = net(images)[0]
            elif mode == 'base':
                preds = net(images)
            preds = (preds > 0.5).float()
            
            num_correct, num_pixels, iou, dice = eval_metrics(preds, masks)
            total_correct += num_correct
            total_pixels += num_pixels
            iou_score.append(iou)
            dice_score.append(dice)
            
            
            # print some information through testing
            if i%5==0:
                print(f'{i}th batch====>>pixel_acc:{num_correct/num_pixels*100:.2f} | iou_score:{iou.mean():.2f} | dice_score:{dice.mean():.2f}')
                
                # _segment_image=masks[0]
                # _out_image=preds[0] 
                
                # img=torch.stack([_segment_image,_out_image],dim=0)
                # save_image(img,f'{save_path}/{i}.png')
            

    print(f"Got {total_correct}/{total_pixels} with acc on test set {total_correct/total_pixels*100:.2f}")
    print(f'Got IoU score on test set: {torch.cat(iou_score, dim=0).mean():.3f}')
    
    print(f'Got Dice score on test set: {sum(dice_score)/len(dice_score):.3f}')

# Function: save checkpoint model
def save_checkpoint(model,optimizer,filepath):
    print("====> Saving checkpoint")
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
    torch.save(checkpoint, filepath)

# Function: load checkpoint model
def load_checkpoint(model,optimizer,filepath):
    print("====> Loading checkpoint")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

# Function: Plot figure
def plot_figure(history, save_figure_path, model='mtl'):
    if model == 'base':
        train_loss = history['train_loss'] 
    elif model =='mtl':
        train_loss = history['train_loss_total']
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    val_iou = history['val_iou']
    num_epoch = range(len(train_loss))
    plt.figure(figsize=(15,6), dpi=100)
    plt.subplot(1,2,1)
    plt.plot(num_epoch, train_loss, label = 'training loss')
    if val_loss:
        plt.plot(num_epoch, val_loss, label = 'validation loss')
    plt.grid(True)
    plt.title('Loss versus epoch number')
    plt.legend()
    if val_loss:
        print('true')
        plt.subplot(1,2,2)
        plt.plot(num_epoch, val_acc, label = 'pixel accuracy')
        plt.plot(num_epoch, val_iou, label = 'IoU score')
        plt.grid(True)
        plt.title('Validation accuracy versus epoch number')
        plt.legend()
    
    plt.savefig(save_figure_path)
    
    

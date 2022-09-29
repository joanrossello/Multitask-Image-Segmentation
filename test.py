"""
Author: Group work
Date: 2021/12/29
Description: utility functions for baseline model 
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
from utils import eval_metrics, load_checkpoint

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


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # initialize
    batch_size = 5
    img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # get test dataset and dataloader
    testset = PetDataset('datasets-oxpet/test/')
    test_generator = PetDatasetGenerator(testset, batch_size, shuffle=True)
    

    
    # load network model
    net=UNET_MTL(3,1).double().to(device)
    load_checkpoint(net,None,'E:/comp0090-cw2/models_weight/mtl_recons_c_epoch_23.pt')
    
    # start testing
    start_time = time.time()
    test(net,device, test_generator)
    # print time spent
    time_spent = time.time() - start_time
    print('Testing complete in {:.0f}m {:.0f}s\n'.format(time_spent // 60 , time_spent % 60))
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import optimizer

from models.unet_mtl_rec import UNET_MTL
from loader import PetDataset, PetDatasetGenerator
from uncertainty_loss import  MTL_uncertain


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

epochs = 50
lr = 1e-4
batch_size = 2

model_unet = UNET_MTL(in_ch=3, out_ch=1).double().to(device)

## task_flags indicate which tasks we are doing in order : seg, class, recon
model = MTL_uncertain(model=model_unet, task_flags=[True, True, True], task_order=[0,2,1])
optimizer = optim.Adam(model.parameters(), lr)

model.train()


trainset = PetDataset('datasets-oxpet/train/')
train_generator = PetDatasetGenerator(trainset, batch_size, shuffle=True)

img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

print('Beginning training...')
for epoch in range(epochs):
    train_generator = PetDatasetGenerator(trainset, batch_size, shuffle=True)
    running_seg_loss = 0.0
    for i, data in enumerate(train_generator):
        images, masks, labels, _ = data
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        images = img_transform(images)
        optimizer.zero_grad()

        (pred_seg, pred_class, pred_recon,
        seg_loss, class_loss, recon_loss,
        logsigmas, loss) = model(images, masks, labels)


        running_seg_loss += seg_loss

        if i % 20 == 0:
            print(f'{epoch}-{i}: TOTAL LOSS = {loss} | SEG LOSS = {seg_loss} | CLASS LOSS = {class_loss} | RECON LOSS = {recon_loss}')
            print(f'\t SEG LOGVAR = {logsigmas[0]} | CLASS LOGVAR = {logsigmas[1]} | RECON LOGVAR = {logsigmas[2]}')
        loss.backward()
        optimizer.step()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_seg_loss,
            }, f'unet_mtl_rec_loss2_{epoch + 1}.pt')
    print(f'EPOCH {epoch} RUNNING SEG LOSS = {running_seg_loss}')
print('Training done')

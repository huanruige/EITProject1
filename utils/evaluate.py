import torch
from torch import nn
from tqdm import tqdm


def eval_BCE(net, dataloader, device):
    # 验证模式
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred = net(image)
            mask_pred = mask_pred.squeeze()
            criteria = nn.CrossEntropyLoss()
            dice_score += criteria(mask_pred, mask_true)

    net.train()
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    MSE_score = 0
    loss_function = nn.MSELoss()
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image = batch['image']
        mask_true = batch['mask']
        mask_true = mask_true.squeeze()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred = net(image)
            MSE_score += loss_function(mask_pred, mask_true)
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return MSE_score
    return MSE_score / num_val_batches



import argparse
import logging
import os.path
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models import VGG16
from models import ECA_ResNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.evaluate import evaluate


project_name = 'two_objects_dif'
# 训练数据集位置
dir_img = Path('E:/GHR/刘金行代码整理/train_data/'+project_name+'/BV/')
dir_mask = Path('E:/GHR/刘金行代码整理/train_data/'+project_name+'/DDL/')
# 训练生成的pth文件保存位置
model_path = '../save_model/'+project_name+'/VGG16'
# model_path = '../save_model/'+project_name+'/ECA_ResNet101'
# model_path = '../save_model/'+project_name+'/ECA_ResNet101'
if not os.path.exists(model_path):
    os.makedirs(model_path)
dir_checkpoint = Path(model_path)


def output_to_image(masks_pred):
    probs = F.softmax(masks_pred, dim=1)[0]
    probs = probs.cpu().squeeze()

    mask = F.one_hot(probs.argmax(dim=0), 3).permute(2, 0, 1).numpy()
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    # 创建数据集
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # 记录最好系数
    best_dice = 20000

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                true_masks = true_masks.squeeze()

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    # 网络输出
                    masks_pred = net(images)
                    # 计算损失
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        val_score = evaluate(net, val_loader, device)
        scheduler.step(val_score)
        logging.info('Validation MSE score: {}'.format(val_score))
        logging.info('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
        # 保存模型
        if val_score < best_dice:
            best_dice = val_score

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # 拼接路径
            torch.save(net, str(dir_checkpoint / 'Best_dice_20.pth'))
            logging.info('Best_dice:{} is saved!'.format(best_dice))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # default=10 默认10个epochs 可修改20-150 eca_resnet要在20以上
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # 训练的网络
    # net = ECA_ResNet.eca_resnet101(num_classes=8982)
    net = VGG16.VGG16(output_size=8982)
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        # 手动停止生成中断文件
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

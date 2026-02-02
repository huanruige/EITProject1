import argparse
import logging
import numpy as np
import pandas
import torch
from utils.data_loading import BasicDataset


def predict_img(net,
                full_img,
                device,
                file_name,
                scale_factor=1,
                out_threshold=0.5,
                ):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        res = output.data.cpu().numpy()
    return res


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    parser.add_argument('--model', '-m', default='.\\save_model\\3Wcircle.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def run_ddl(data_path, file_name, name, save_path, model_path):
    args = get_args()
    file_name = data_path + file_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    # 模型地址
    model_name = model_path
    net = torch.load(model_name,map_location='cpu', weights_only=False)
    net.to(device=device)
    img = pandas.read_csv(file_name, header=None)
    my_array = np.array(img)
    my_tensor = torch.tensor(my_array)

    ddl = predict_img(
        net=net,
        full_img=my_tensor,
        scale_factor=args.scale,
        out_threshold=args.mask_threshold,
        device=device,
        file_name=name,
    )
    return ddl

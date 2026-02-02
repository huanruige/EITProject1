import os
import numpy as np
from utils.predict_ddl_vgg import run_ddl


def get_file_name(file_dir):
    file_name = []
    root_name = []
    for root, dirs, files in os.walk(file_dir):
        root_name.append(root)
        file_name.append(files)
    return file_name, root_name


if __name__ == "__main__":
    path_name = 'two_objects_dif'
    # 验证数据集位置 只需要电压 电导率是用网络预测的
    bv_path = 'E:/GHR/刘金行代码整理/test_data/'+path_name+'/BV/'
    # 预测电导率保存位置
    save_path = '../predict_ddl/' + path_name + '/VGG16/Best_dice_20/'
    # save_path = '../predict_ddl/' + path_name + '/ECA_ResNet101/Best_dice_30/'
    if not os.path.exists(save_path):  # 检查目录是否存在
        os.makedirs(save_path)  # 如果不存在则创建目录

    # 获取文件夹下的所有文件名
    file_name, root = get_file_name(bv_path)

    file_name = file_name[0]
    print(file_name)

    model_ddl_path = '../save_model/' + path_name + '/VGG16/Best_dice_20.pth'
    # model_ddl_path = '../save_model/' + path_name + '/ECA_ResNet101/Best_dice_30.pth'
    print("测试样本长度：", len(file_name))
    for i in range(0, len(file_name)):
        name = file_name[i][:-4].replace('_bv', '_ddl')
        ddl = run_ddl(bv_path, file_name[i], name, save_path, model_ddl_path)
        np.savetxt(save_path + name + '.csv', ddl, fmt="%f", delimiter=',')
        print("第{}个样本的电导率预测完成并保存".format(i + 1))
    print("预测完成 使用matlab生成图像")

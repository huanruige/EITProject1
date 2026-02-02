文件结构
models：模型文件，有VGG16，ECA_ResNet18/34/50/101/152，VGG16和ECA_ResNet101可用，其他没试过
predict：预测脚本
predict_ddl：存放预测脚本生成的预测电导率
save_model：存放训练脚本生成的pth文件
train：训练脚本
utils：工具类，其中data_loading用于读入数据集

运行流程
0. 配置Python环境，Matlab+Eidors环境，修改路径
1. 使用数据生成代码guizewuti.m，生成训练数据集train_data，验证数据集test_data
2. 运行train_8982进行训练
3. 运行predict_8982进行预测
4. 使用成像代码colorbar_to_image.m进行成像，成像结果保存在images文件夹下
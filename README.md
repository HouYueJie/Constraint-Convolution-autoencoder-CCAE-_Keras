# Constraint-Convolution-autoencoder-CCAE-_Keras
The data is compressed by using the constraint convolution autoencoder.


# 一. 搭建环境
## -环境配置
numpy==1.19.4 
tensorflow-gpu==2.4.0 
tqdm==4.54.0 
matplotlib==3.2.1 
opencv-python==4.4.0.46 
Keras==2.4.3 
Pillow==7.1.2 
h5py==2.10.0 

# 二. 配置 config.py
## -路径配置
ccae_result_path <=> 模型训练结果保存地址
save_encode_data <=> 模型中间层数据保存地址

## -网络层配置
model_config <=> 可配置配置embedding层，convolution层，deconvolution层，pooling层，upsampling层，connection层。【具体配置可参考模板】

## -训练配置
train_config <=> 可配置批次大小，保存步长，模型参数保存地址，学习率，训练轮次，训练数据地址。

## -预测配置
predict_config <=> 可配置预测的batch size

# 三. 训练模型
GPU运行 python3 main.py train gpu [多GPU情况，具体运行看设备号。main.py 中 os.environ["CUDA_VISIBLE_DEVICES"] 可按照GPU序号，更换GPU设备运行。]
CPU运行 python3 main.py train cpu
根据提示，选择重新训练或者继承权重继续训练。

# 四. 查看训练效果【loss和重构图像】
进入 result
运行 python3 generate_image.py

# 五. 提取中间层数据
GPU运行 python3 main.py encode gpu [多GPU情况，具体运行看设备号。main.py 中 os.environ["CUDA_VISIBLE_DEVICES"] 可按照GPU序号，更换GPU设备运行。]
CPU运行 python3 main.py encode cpu

根据提示选择需要的中间层，获取中间层数据。

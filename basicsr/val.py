import torch
import os
from basicsr.models import create_model
from basicsr.utils import mkdir_and_rename
from basicsr.data import create_dataloader, create_dataset
from basicsr.utils.options import parse
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img  # 引入imwrite和tensor2img

def load_model(opt, checkpoint_path):
    # 创建模型
    model = create_model(opt)
    # 使用 load_network 方法加载模型权重
    model.load_network(model.net_g, checkpoint_path, strict=False, param_key='params')
    # 设置模型为评估模式
    model.net_g.eval()  # 使用 net_g 作为实际网络调用 eval()
    return model

def validate(opt, model, val_loader):
    # 创建保存结果的文件夹
    save_img = opt['val'].get('save_img', True)  # 修改为True以便保存图像
    save_dir = opt['val'].get('save_dir', './results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 验证过程
    with torch.no_grad():
        for idx, val_data in tqdm(enumerate(val_loader), desc="Validating"):
            model.feed_data(val_data)  # 将数据输入到模型中
            model.test()  # 进行推理

            # 获取模型输出
            visuals = model.get_current_visuals()  # 获取输出的图像等
            output_img = visuals['result']  # 这里假设输出图像存储在 'result' 中，具体依赖于你的模型

            # 保存结果图像
            for i, img in enumerate(output_img):
                img_name = val_data['image_name'][0] + '.png'
                sr_img = tensor2img([output_img])  # 转换为图像格式
                img_path = os.path.join(save_dir, img_name)
                imwrite(sr_img, img_path)  # 使用模型的保存函数来保存图片

            print(f"Processed {idx + 1} batches, saved images to {save_dir}")

def main():
    # 加载配置文件
    opt = parse('../options/train/HighREV/EFNet_HighREV_Deblur.yml', is_train=False)

    # 手动设置 dist 为 False，避免分布式训练相关的错误
    opt['dist'] = False

    # 加载训练好的模型
    checkpoint_path = '../experiments/EFNet_highrev_single_deblur/models/net_g_20000.pth'  # 这里是保存的模型路径
    model = load_model(opt, checkpoint_path)

    # 创建验证集的 DataLoader
    val_set = create_dataset(opt['datasets']['val'])
    val_loader = create_dataloader(
        val_set, opt['datasets']['val'], num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed']
    )

    # 进行验证并保存结果
    validate(opt, model, val_loader)

if __name__ == "__main__":
    main()

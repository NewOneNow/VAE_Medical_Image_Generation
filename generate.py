import torch
from models.vae import VAE
import matplotlib.pyplot as plt
import os

def generate():
    # 参数设置
    latent_dim = 128
    img_channels = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = VAE(img_channels=img_channels, latent_dim=latent_dim).to(device)
    checkpoint = 'E:\\VAEMIG\\checkpoints\\vae_final.pth'  # 替换为您的模型路径
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    # 从潜在空间采样并生成图像
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z).cpu()
        # 可视化生成的图像
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        for i in range(samples.size(0)):
            plt.imshow(samples[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.savefig(f'outputs/sample_{i}.png')
            plt.close()
        print('生成的图像已保存到 outputs/ 目录中。')

if __name__ == "__main__":
    generate()

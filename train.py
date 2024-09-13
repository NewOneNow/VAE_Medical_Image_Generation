# train.py
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.vae import VAE
from data.dataset import MedicalImageDataset
from utils.utils import loss_function, EarlyStopping
import os
import matplotlib.pyplot as plt
import time

def train():
    # 参数设置
    data_dir = 'E:\\VAEMIG\\datasets\\processed_data'  # 预处理后数据的路径
    epochs = 100  # 增加 epochs，以便观察早停
    batch_size = 8
    learning_rate = 1e-3
    img_channels = 1
    latent_dim = 128
    patience = 10  # 早停的耐心次数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = MedicalImageDataset(data_dir, transform=transform)
    # 划分训练集和验证集，80% 训练，20% 验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型和优化器
    model = VAE(img_channels=img_channels, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 添加用于记录损失的列表
    train_losses = []
    val_losses = []

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss = 0
        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_dataloader)}] Loss: {loss.item()/len(data):.4f}')
        average_train_loss = train_loss / len(train_dataset)
        train_losses.append(average_train_loss)

        # 在验证集上评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        average_val_loss = val_loss / len(val_dataset)
        val_losses.append(average_val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'====> Epoch: {epoch+1} Train loss: {average_train_loss:.4f} Val loss: {average_val_loss:.4f} Time: {epoch_duration:.2f}s')

        # 检查早停条件
        early_stopping(average_val_loss, model)

        # 保存当前模型
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(model.state_dict(), f'checkpoints/vae_epoch_{epoch+1}.pth')

        if early_stopping.early_stop:
            print("早停触发，停止训练")
            # 加载验证损失最好的模型参数
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # 在训练结束后，保存损失日志和绘制曲线
    save_loss_log(train_losses, val_losses)
    plot_loss_curve(train_losses, val_losses)

    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/vae_final.pth')

def save_loss_log(train_losses, val_losses):
    # 将损失值保存到 loss_log.txt 文件
    with open('loss_log.txt', 'w') as f:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}\n')
    print('损失日志已保存到 loss_log.txt')

def plot_loss_curve(train_losses, val_losses):
    # 绘制损失曲线并保存为 loss_curve.png
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()
    print('损失曲线已保存为 loss_curve.png')

if __name__ == "__main__":
    train()

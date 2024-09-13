# utils/utils.py
import torch.nn.functional as F
import torch

def loss_function(recon_x, x, mu, logvar):
    # 重构损失（使用均方误差）
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KL 散度损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

class EarlyStopping:
    """早停机制，当验证损失在设定的 patience（耐心）次数内没有降低时，停止训练"""
    def __init__(self, patience=8, verbose=False, delta=0):
        """
        Args:
            patience (int): 当验证损失在多少个 epoch 内没有提升时，停止训练
            verbose (bool): 如果为 True，则每次验证损失更新时打印信息
            delta (float): 验证损失的提升阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.4f}. Saving model...')
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.4f}. Saving model...')
            self.counter = 0

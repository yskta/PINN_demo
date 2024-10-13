import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.workbook.protection import WorkbookProtection
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import os

import torch
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
import time
import logging
from datetime import datetime

# 現在の日時を取得
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# ログファイル名に日時を含める
log_filename = f'training_log_{current_time}.txt'

logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# 時間かかる
#P.3    温度測定位置
# x：0~90mm, y：0~90mm, z：0~500mm

# Caution：本来は100分割すべきだが、クラッシュするので50分割にしている
# NumPy の配列（ndarray）
x = np.linspace(0.0, 0.09, 45, dtype=np.float32)[:, None]
y = np.linspace(0.0, 0.09, 45, dtype=np.float32)[:, None]
z = np.linspace(0.0, 0.5, 50, dtype=np.float32)[:, None]
t_original = np.linspace(0, 118800, 1980, dtype=np.int32)[:, None]

# 60秒（1分）単位で四捨五入
t_rounded = np.round(t_original / 60) * 60

# int32 型に変換
t = t_rounded.astype(np.int32)

# メッシュグリッドの作成
X, Y, Z, T = np.meshgrid(x.flatten(), y.flatten(), z.flatten(), t.flatten(), indexing='ij')  #(45, 45, 50, 1980)

# 構造化配列の定義
dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('t', np.int32)])

# グリッドの作成
grid = np.empty((45, 45, 50, 1980), dtype=dt)
grid['x'], grid['y'], grid['z'], grid['t'] = X, Y, Z, T

# 円の中にある点のマスクを作成
circle_mask = (grid['x']**2 + grid['y']**2) <= 0.09**2

# マスクを使って、条件を満たさない点を-1で埋める
grid_inside = np.where(circle_mask, grid, np.full(grid.shape, -1, dtype=dt))

# -1でない要素の数を確認（実際に条件を満たす点の数）
non_negative_count = np.count_nonzero(grid_inside['x'] != -1)

# 必要に応じて、-1でない要素のみを取り出す
grid_target = grid_inside[grid_inside['x'] != -1]

# 境界条件の関数を定義
def is_boundary(x, y, z):
    return np.isclose(z, 0.5) | np.isclose(x**2 + y**2, 0.09**2, atol=1e-6)

# データを分類
ic_mask = (grid_target['t'] == 0) & is_boundary(grid_target['x'], grid_target['y'], grid_target['z'])
bc_mask = (grid_target['t'] > 0) & is_boundary(grid_target['x'], grid_target['y'], grid_target['z'])
interior_mask = ~(ic_mask | bc_mask)

# データを抽出
IC = grid_target[ic_mask]
BC = grid_target[bc_mask]
Interior = grid_target[interior_mask]

# ICとBCのデータを4列の2D配列に変換
IC_reshaped = np.column_stack((IC['x'], IC['y'], IC['z'], IC['t']))
BC_reshaped = np.column_stack((BC['x'], BC['y'], BC['z'], BC['t']))
Interior_reshaped = np.column_stack((Interior['x'], Interior['y'], Interior['z'], Interior['t']))

# 残差を取る点のランダムサンプリング
# ICは全部使う(1662個のデータ)
# TODO：サンプリング数を変更する場合は、ここを変更。変数化しても良い。
IC_sampled = IC_reshaped
# BC：10000
BC_sampled = BC_reshaped[np.random.choice(BC_reshaped.shape[0], 10000, replace=False)]
# Interior：30000
Interior_sampled = Interior_reshaped[np.random.choice(Interior_reshaped.shape[0], 30000, replace=False)]

# 初期条件の温度
u_0 = np.zeros((IC_sampled.shape[0], 1))
u_0[:, 0] = 18.3

# P.1   計算前提
# 放射率
ε = 1.0
# Stefan-Boltzmann constant
sigma = 5.67e-8 # W/m^2/K^4

# P.2   伝熱方程式と物性値
# 密度
rho = 1404.0 # kg/m^3

lambda_f, lambda_IC, lambda_BC = 1.0, 1.0, 1.0
layers = [4, 50, 50, 50, 50, 1]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.ELU
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, xyzt):
        out = self.layers(xyzt)
        return out

class PhysicsInformedNN():
    def __init__(self, IC, BC, Interior, layers, u_0, lambda_f, lambda_IC, lambda_BC, rho, ε, sigma, batch_size):
        self.xyz_IC = torch.tensor(IC[:, 0:3], requires_grad=True).float().to(device)
        self.t_IC = torch.tensor(IC[:, 3:4], requires_grad=True).float().to(device)
        self.xyz_BC = torch.tensor(BC[:, 0:3], requires_grad=True).float().to(device)
        self.t_BC = torch.tensor(BC[:, 3:4], requires_grad=True).float().to(device)
        self.xyz_Interior = torch.tensor(Interior[:, 0:3], requires_grad=True).float().to(device)
        self.t_Interior = torch.tensor(Interior[:, 3:4], requires_grad=True).float().to(device)
        self.layers = layers
        self.u_0 = torch.tensor(u_0, dtype=torch.float32).to(device)
        self.rho = rho
        self.ε = ε
        self.sigma = sigma
        self.batch_size = batch_size
        self.dnn = DNN(layers).to(device)
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-4)
        self.lambda_f = torch.tensor(lambda_f, requires_grad=True, device=device)
        self.lambda_IC = torch.tensor(lambda_IC, requires_grad=True, device=device)
        self.lambda_BC = torch.tensor(lambda_BC, requires_grad=True, device=device)

    def net_u(self, xyz, t):
        # if x.dim() == 1:
            # x = x.unsqueeze(1)
        # if t.dim() == 1:
            # t = t.unsqueeze(1)
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]
        u = self.dnn(torch.cat([x, y, z, t], 1))
        return u
    
    # 物性値
    # 比熱：温度に依存。# J/(kg・K)
    def cp(self, xyz, t):
        u = self.net_u(xyz, t)
        return torch.where(
            u <= 25, 760.0,
            torch.where(u <= 200, 760.0 + (u-25) * (995.0-760.0) / 175.0,
            torch.where(u <= 400, 995.0 + (u-200) * (1100.0-995.0) / 200.0,
            torch.where(u <= 600, 1100.0 + (u-400) * (1150.0-1100.0) / 200.0,
            torch.where(u <= 800, 1150.0 + (u-600) * (1190.0-1150.0) / 200.0,
            torch.where(u <= 1000, 1190.0 + (u-800) * (1230.0-1190.0) / 200.0,
            torch.where(u <= 1200, 1230.0 + (u-1000) * (1290.0-1230.0) / 200.0,
            1290.0))))))) 
    
    def λ_x_y(self, xyz, t):
        u = self.net_u(xyz, t)
        return torch.where(
            u <= 25, 1.43,
            torch.where(u <= 200, 1.43 + (u-25) * (1.05-1.43) / 175.0,
            torch.where(u <= 400, 1.05 + (u-200) * (0.81-1.05) / 200.0,
            torch.where(u <= 600, 0.81 + (u-400) * (0.68-0.81) / 200.0,
            torch.where(u <= 800, 0.68 + (u-600) * (0.59-0.68) / 200.0,
            torch.where(u <= 1000, 0.59 + (u-800) * (0.54-0.59) / 200.0,
            torch.where(u <= 1200, 0.54 + (u-1000) * (0.53-0.54) / 200.0,
            0.53)))))))
        
    def λ_z(self, xyz, t):
        u = self.net_u(xyz, t)
        return torch.where(
            u <= 25, 2.60,
            torch.where(u <= 200, 2.60 + (u-25) * (1.92-2.60) / 175.0,
            torch.where(u <= 400, 1.92 + (u-200) * (1.48-1.92) / 200.0,
            torch.where(u <= 600, 1.48 + (u-400) * (1.24-1.48) / 200.0,
            torch.where(u <= 800, 1.24 + (u-600) * (1.08-1.24) / 200.0,
            torch.where(u <= 1000, 1.08 + (u-800) * (0.99-1.08) / 200.0,
            torch.where(u <= 1200, 0.99 + (u-1000) * (0.96-0.99) / 200.0,
            0.96)))))))
        
    def boundary_Temperature(self, t):
        return 20.0 + 30.0 / 3600.0 * t + 273.15

    # PDEの残差
    def net_f(self, xyz, t):
        u = self.net_u(xyz, t)
        c = self.cp(xyz, t)
        λ_x_y = self.λ_x_y(xyz, t)
        λ_z = self.λ_z(xyz, t)
        
        # after
        # 各方向の勾配を計算
        grads = torch.autograd.grad(u.sum(), [t, xyz], create_graph=True)
        u_t, u_xyz = grads[0], grads[1]
        
        # 二階微分の計算
        u_xx = torch.autograd.grad(u_xyz[:, 0].sum(), xyz, create_graph=True)[0][:, 0].unsqueeze(1)
        u_yy = torch.autograd.grad(u_xyz[:, 1].sum(), xyz, create_graph=True)[0][:, 1].unsqueeze(1)
        u_zz = torch.autograd.grad(u_xyz[:, 2].sum(), xyz, create_graph=True)[0][:, 2].unsqueeze(1)

        f = rho * c * u_t - (λ_x_y * u_xx + λ_x_y * u_yy + λ_z * u_zz)
        return f
    
    # 初期条件
    def net_IC(self, xyz, t):
        u = self.net_u(xyz, t)
        IC = self.u_0 - u
        return IC
    
    # 境界条件
    def net_BC(self, xyz, t):
        T_inf = self.boundary_Temperature(t)
        u = self.net_u(xyz, t) + 273.15
        Q = self.ε * self.sigma * (T_inf**4 - u**4)
        λ_x_y = self.λ_x_y(xyz, t)
        λ_z = self.λ_z(xyz, t)
        
        # after
        grads = torch.autograd.grad(u.sum(), xyz, create_graph=True)[0]
        u_x = grads[:, 0].unsqueeze(1)
        u_y = grads[:, 1].unsqueeze(1)
        u_z = grads[:, 2].unsqueeze(1)

        BC = Q + (λ_x_y * u_x + λ_x_y * u_y + λ_z * u_z)
        return BC
    
    # 損失関数
    def loss_func(self):
        f_pred = self.net_f(self.xyz_Interior, self.t_Interior)
        IC_pred = self.net_IC(self.xyz_IC, self.t_IC)
        BC_pred = self.net_BC(self.xyz_BC, self.t_BC)
        loss_f = torch.mean(f_pred**2)
        loss_IC = torch.mean(IC_pred**2)
        loss_BC = torch.mean(BC_pred**2)
        total_loss = self.lambda_f * loss_f + self.lambda_IC * loss_IC + self.lambda_BC * loss_BC
        return total_loss, loss_f, loss_IC, loss_BC

    def train(self, nEpochs):
        for epoch in range(nEpochs):
            self.optimizer_Adam.zero_grad()
            total_loss, loss_f, loss_IC, loss_BC = self.loss_func()
            total_loss.backward()
            self.optimizer_Adam.step()
            with torch.no_grad():
                self.lambda_f *= torch.exp(0.1 * loss_f.detach() / total_loss.detach())
                self.lambda_IC *= torch.exp(0.1 * loss_IC.detach() / total_loss.detach())
                self.lambda_BC *= torch.exp(0.1 * loss_BC.detach() / total_loss.detach())
            if epoch % 100 == 0 or epoch == nEpochs - 1:
                log_message = f'Epoch {epoch}: Total Loss: {total_loss.item():.4e}, Loss f: {loss_f.item():.4e}, Loss IC: {loss_IC.item():.4e}, Loss BC: {loss_BC.item():.4e}'
                logging.info(log_message)
                log_message = f'lambda_f: {self.lambda_f.item()}, lambda_IC: {self.lambda_IC.item()}, lambda_BC: {self.lambda_BC.item()}'
                logging.info(log_message)

    def predict(self, X_star):
        xyz_star = torch.tensor(X_star[:, 0:3], requires_grad=True).float().to(device)
        t_star = torch.tensor(X_star[:, 3:4], requires_grad=True).float().to(device)
        u_pred = self.net_u(xyz_star, t_star)
        return u_pred.detach().cpu().numpy()

# モデルのインスタンス化
model = PhysicsInformedNN(IC=IC_sampled, BC=BC_sampled, Interior=Interior_sampled, layers=layers, u_0=u_0, lambda_f=lambda_f, lambda_IC=lambda_IC, lambda_BC=lambda_BC, rho=rho, ε = ε, sigma = sigma, batch_size=150)

model.train(100)

# grid_targetをreshape
X_star = np.column_stack((grid_target['x'], grid_target['y'], grid_target['z'], grid_target['t']))

u_pred = model.predict(X_star)
U_pred = griddata(X_star[:, 0:4], u_pred.flatten(), (X, Y, Z, T), method='linear')

def plot_3d_isosurface(U_pred, time_index, time_value):
    x_min, x_max = 0.0, 0.09
    y_min, y_max = 0.0, 0.09
    z_min, z_max = 0.0, 0.5
    
    fig = plt.figure(figsize=(12, 10))
     # 等温面のサブプロット
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    temp_thresholds = [18, 200, 400, 600, 800, 1000]
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'purple']

    for threshold, color in zip(temp_thresholds, colors):
        verts, faces, _, _ = measure.marching_cubes(U_pred[:,:,:,time_index], threshold)
        
        verts[:, 0] = x_min + verts[:, 0] * (x_max - x_min) / (U_pred.shape[0] - 1)
        verts[:, 1] = y_min + verts[:, 1] * (y_max - y_min) / (U_pred.shape[1] - 1)
        verts[:, 2] = z_min + verts[:, 2] * (z_max - z_min) / (U_pred.shape[2] - 1)
        
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                                color=color, alpha=0.3, shade=True)

    # カラーバーを追加
    m = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
    m.set_array(temp_thresholds)
    cbar = plt.colorbar(m, ax=ax, label='Temperature [°C]', pad=0.1)
    cbar.set_ticks(temp_thresholds)

    plt.tight_layout()
    plt.show()

    # figディレクトリを作成（存在しない場合）
    if not os.path.exists('fig'):
        os.makedirs('fig')
    
    # 図を保存
    plt.savefig(f'fig/temperature_distribution_{time_value}s.png', dpi=300, bbox_inches='tight')
    plt.close()  # メモリを節約するためにfigureを閉じる

# 使用例
time_steps = [0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 118800]

for time_step in time_steps:
    time_index = np.where(t.flatten() == time_step)[0][0]
    plot_3d_isosurface(U_pred, time_index, time_step)
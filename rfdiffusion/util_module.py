import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import copy
import dgl
from rfdiffusion.util import base_indices, RTs_by_torsion, xyzs_in_base_frame, rigid_from_3_points

# 初始化LeCun正态分布的模块权重
def init_lecun_normal(module):
    # 生成截断正态分布的样本
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        # 创建一个标准正态分布
        normal = torch.distributions.normal.Normal(0, 1)
        # 计算截断的边界
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        # 计算截断点的累积分布函数值
        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        # 生成截断后的概率值
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform
        # 对概率值进行裁剪
        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        # 根据逆误差函数生成截断正态分布的值
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        # 对生成的值进行裁剪
        x = torch.clamp(x, a, b)
        return x

    # 采样截断正态分布
    def sample_truncated_normal(shape):
        # 计算标准差
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    # 将采样得到的权重赋值给模块的权重参数
    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

# 初始化LeCun正态分布的参数权重
def init_lecun_normal_param(weight):
    # 生成截断正态分布的样本
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        # 创建一个标准正态分布
        normal = torch.distributions.normal.Normal(0, 1)
        # 计算截断的边界
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        # 计算截断点的累积分布函数值
        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        # 生成截断后的概率值
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform
        # 对概率值进行裁剪
        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        # 根据逆误差函数生成截断正态分布的值
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        # 对生成的值进行裁剪
        x = torch.clamp(x, a, b)
        return x

    # 采样截断正态分布
    def sample_truncated_normal(shape):
        # 计算标准差
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    # 将采样得到的权重赋值给参数权重
    weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
    return weight

# 用于梯度检查点的自定义前向传播函数
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

# 复制模块N次
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 自定义的Dropout层，可对整行或整列进行丢弃
class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # 定义一个伯努利分布采样器，以1-p_drop的概率采样1
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop

    def forward(self, x):
        # 在评估模式下不进行Dropout
        if not self.training: 
            return x
        # 获取输入的形状
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        # 采样得到掩码
        mask = self.sampler.sample(shape).to(x.device).view(shape)
        # 应用掩码并进行缩放
        x = mask * x / (1.0 - self.p_drop)
        return x

# 距离径向基函数
def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    # 生成均匀分布的中心值
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    # 计算标准差
    D_sigma = (D_max - D_min) / D_count
    # 扩展维度
    D_expand = torch.unsqueeze(D, -1)
    # 计算径向基函数的值
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

# 计算序列分离特征
def get_seqsep(idx):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    # 计算序列分离矩阵
    seqsep = idx[:,None,:] - idx[:,:,None]
    # 获取符号
    sign = torch.sign(seqsep)
    # 获取绝对值
    neigh = torch.abs(seqsep)
    # 将非相邻残基的距离置为0
    neigh[neigh > 1] = 0.0 
    # 恢复符号
    neigh = sign * neigh
    return neigh.unsqueeze(-1)

# 构建全连接图
def make_full_graph(xyz, pair, idx, top_k=64, kmin=9):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # 计算序列分离矩阵
    sep = idx[:,None,:] - idx[:,:,None]
    # 获取非相邻残基的索引
    b,i,j = torch.where(sep.abs() > 0)
   
    # 计算源节点和目标节点的索引
    src = b*L+i
    tgt = b*L+j
    # 创建图
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    # 计算相对位置并赋值给边的属性
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() 

    return G, pair[b,i,j][...,None]

# 构建Top-K图
def make_topk_graph(xyz, pair, idx, top_k=64, kmin=32, eps=1e-6):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # 计算当前CA坐标的距离矩阵
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # 计算序列分离矩阵
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    # 结合距离和序列分离信息
    D = D + sep*eps
    
    # 获取每个节点的Top-K邻居
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) 
    topk_matrix = torch.zeros((B, L, L), device=device)
    # 标记Top-K邻居
    topk_matrix.scatter_(2, E_idx, 1.0)

    # 满足以下条件之一则添加边：
    #   1) |i-j| <= kmin (连接相邻的残基)
    #   2) Top-K邻居
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = torch.where(cond)
   
    # 计算源节点和目标节点的索引
    src = b*L+i
    tgt = b*L+j
    # 创建图
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    # 计算相对位置并赋值给边的属性
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() 

    return G, pair[b,i,j][...,None]

# 生成绕X轴旋转的旋转矩阵
def make_rotX(angs, eps=1e-6):
    B,L = angs.shape[:2]
    # 计算向量的范数
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    # 初始化旋转矩阵
    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    # 赋值旋转矩阵的元素
    RTs[:,:,1,1] = angs[:,:,0]/NORM
    RTs[:,:,1,2] = -angs[:,:,1]/NORM
    RTs[:,:,2,1] = angs[:,:,1]/NORM
    RTs[:,:,2,2] = angs[:,:,0]/NORM
    return RTs

# 生成绕Z轴旋转的旋转矩阵
def make_rotZ(angs, eps=1e-6):
    B,L = angs.shape[:2]
    # 计算向量的范数
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    # 初始化旋转矩阵
    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    # 赋值旋转矩阵的元素
    RTs[:,:,0,0] = angs[:,:,0]/NORM
    RTs[:,:,0,1] = -angs[:,:,1]/NORM
    RTs[:,:,1,0] = angs[:,:,1]/NORM
    RTs[:,:,1,1] = angs[:,:,0]/NORM
    return RTs

# 生成绕任意轴旋转的旋转矩阵
def make_rot_axis(angs, u, eps=1e-6):
    B,L = angs.shape[:2]
    # 计算向量的范数
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    # 初始化旋转矩阵
    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    # 计算余弦和正弦值
    ct = angs[:,:,0]/NORM
    st = angs[:,:,1]/NORM
    # 获取轴向量的分量
    u0 = u[:,:,0]
    u1 = u[:,:,1]
    u2 = u[:,:,2]

    # 赋值旋转矩阵的元素
    RTs[:,:,0,0] = ct+u0*u0*(1-ct)
    RTs[:,:,0,1] = u0*u1*(1-ct)-u2*st
    RTs[:,:,0,2] = u0*u2*(1-ct)+u1*st
    RTs[:,:,1,0] = u0*u1*(1-ct)+u2*st
    RTs[:,:,1,1] = ct+u1*u1*(1-ct)
    RTs[:,:,1,2] = u1*u2*(1-ct)-u0*st
    RTs[:,:,2,0] = u0*u2*(1-ct)-u1*st
    RTs[:,:,2,1] = u1*u2*(1-ct)+u0*st
    RTs[:,:,2,2] = ct+u2*u2*(1-ct)
    return RTs

# 计算所有原子的坐标
class ComputeAllAtomCoords(nn.Module):
    def __init__(self):
        super(ComputeAllAtomCoords, self).__init__()

        # 初始化基础索引
        self.base_indices = nn.Parameter(base_indices, requires_grad=False)
        # 初始化基础框架中的旋转矩阵
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        # 初始化基础框架中的坐标
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False)

    def forward(self, seq, xyz, alphas, non_ideal=False, use_H=True):
        B,L = xyz.shape[:2]

        # 从三个点计算旋转矩阵和平移向量
        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], non_ideal=non_ideal)

        # 初始化基础旋转矩阵
        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)

        # 赋值旋转矩阵和平移向量
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts

        # 计算omega角的旋转矩阵
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq,0,:], make_rotX(alphas[:,:,0,:]))

        # 计算phi角的旋转矩阵
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,1,:], make_rotX(alphas[:,:,1,:]))

        # 计算psi角的旋转矩阵
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,2,:], make_rotX(alphas[:,:,2,:]))

        # 计算CB弯曲的旋转轴
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5*(basexyzs[:,:,2,:3]+basexyzs[:,:,0,:3])
        CAr = (basexyzs[:,:,1,:3])
        CBr = (basexyzs[:,:,4,:3])
        CBrotaxis1 = (CBr-CAr).cross(NCr-CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True)+1e-8
        
        # 计算CB扭转的旋转轴
        NCp = basexyzs[:,:,2,:3] - basexyzs[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCr, dim=-1, keepdim=True)/ torch.sum(NCr*NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr-CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True)+1e-8
        
        # 计算CB弯曲和扭转的旋转矩阵
        CBrot1 = make_rot_axis(alphas[:,:,7,:], CBrotaxis1 )
        CBrot2 = make_rot_axis(alphas[:,:,8,:], CBrotaxis2 )
        
        # 计算最终的CB旋转矩阵
        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, CBrot1,CBrot2)
        
        # 计算chi1和CG弯曲的旋转矩阵
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim', 
            RTF8, 
            self.RTs_in_base_frame[seq,3,:], 
            make_rotX(alphas[:,:,3,:]), 
            make_rotZ(alphas[:,:,9,:]))

        # 计算chi2的旋转矩阵
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF4, self.RTs_in_base_frame[seq,4,:],make_rotX(alphas[:,:,4,:]))

        # 计算chi3的旋转矩阵
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF5,self.RTs_in_base_frame[seq,5,:],make_rotX(alphas[:,:,5,:]))

        # 计算chi4的旋转矩阵
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF6,self.RTs_in_base_frame[seq,6,:],make_rotX(alphas[:,:,6,:]))

        # 堆叠所有旋转矩阵
        RTframes = torch.stack((
            RTF0,RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7,RTF8
        ),dim=2)

        # 计算所有原子的坐标
        xyzs = torch.einsum(
            'brtij,brtj->brti', 
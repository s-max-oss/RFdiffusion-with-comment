import torch.utils.checkpoint as checkpoint
from rfdiffusion.util_module import *
from rfdiffusion.Attention_module import *
from rfdiffusion.SE3_network import SE3TransformerWrapper

# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        # 继承自 nn.Module，用于更新 MSA 特征
        super(MSAPairStr2MSA, self).__init__()
        # 对 Pair 特征进行 Layer Normalization
        self.norm_pair = nn.LayerNorm(d_pair)
        # 将 Pair 特征和 rbf 特征拼接后进行线性变换
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        # 对状态特征进行 Layer Normalization
        self.norm_state = nn.LayerNorm(d_state)
        # 将状态特征投影到 MSA 特征维度
        self.proj_state = nn.Linear(d_state, d_msa)
        # 按行进行 Dropout 操作
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        # 带有偏置的 MSA 行注意力模块
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            # 使用全局注意力的 MSA 列注意力模块
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            # 普通的 MSA 列注意力模块
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        # 前馈层，用于进一步处理 MSA 特征
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # 对参数进行初始化
        self.reset_parameter()

    def reset_parameter(self):
        # 对 proj_pair 和 proj_state 进行 LeCun 正态初始化
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # 将 proj_pair 和 proj_state 的偏置初始化为零
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state):
        '''
        Inputs:
            - msa: MSA 特征 (B, N, L, d_msa)
            - pair: Pair 特征 (B, L, L, d_pair)
            - rbf_feat: Ca-Ca 距离特征，从 xyz 坐标计算得到 (B, L, L, 36)
            - state: SE(3)-Transformer 层更新后的节点特征 (B, L, d_state)
        Output:
            - msa: 更新后的 MSA 特征 (B, N, L, d_msa)
        '''
        # 获取 msa 的批次大小 B、序列数量 N 和序列长度 L
        B, N, L = msa.shape[:3]

        # 对 Pair 特征进行归一化
        pair = self.norm_pair(pair)
        # 将 Pair 特征和 rbf 特征拼接
        pair = torch.cat((pair, rbf_feat), dim=-1)
        # 对拼接后的特征进行线性变换
        pair = self.proj_pair(pair) # (B, L, L, d_pair)
        #
        # 对状态特征进行归一化
        state = self.norm_state(state)
        # 将状态特征投影到 MSA 特征维度，并重塑形状
        state = self.proj_state(state).reshape(B, 1, L, -1)
        # 将状态特征加到 MSA 特征的第一个序列上
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # 对 MSA 特征应用行注意力，并加上 Dropout
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        # 对 MSA 特征应用列注意力
        msa = msa + self.col_attn(msa)
        # 对 MSA 特征应用前馈层
        msa = msa + self.ff(msa)

        return msa

class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        # 继承自 nn.Module，用于更新 Pair 特征
        super(PairStr2Pair, self).__init__()
        
        # 将 rbf 特征嵌入到隐藏维度
        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        # 将隐藏维度的 rbf 特征投影到 Pair 特征维度
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        # 按行进行 Dropout 操作
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        # 按列进行 Dropout 操作
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        # 带有偏置的行轴向注意力模块
        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        # 带有偏置的列轴向注意力模块
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        # 前馈层，用于进一步处理 Pair 特征
        self.ff = FeedForwardLayer(d_pair, 2)
        
        # 对参数进行初始化
        self.reset_parameter()
    
    def reset_parameter(self):
        # 对 emb_rbf 的权重进行 Kaiming 正态初始化
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        # 将 emb_rbf 的偏置初始化为零
        nn.init.zeros_(self.emb_rbf.bias)
        
        # 对 proj_rbf 进行 LeCun 正态初始化
        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        # 将 proj_rbf 的偏置初始化为零
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat):
        # 获取 Pair 特征的批次大小 B 和序列长度 L
        B, L = pair.shape[:2]

        # 对 rbf 特征进行嵌入和投影操作
        rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))

        # 对 Pair 特征应用行注意力，并加上 Dropout
        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat))
        # 对 Pair 特征应用列注意力，并加上 Dropout
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat))
        # 对 Pair 特征应用前馈层
        pair = pair + self.ff(pair)
        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        # 继承自 nn.Module，用于将 MSA 特征转换为 Pair 特征
        super(MSA2Pair, self).__init__()
        # 对 MSA 特征进行 Layer Normalization
        self.norm = nn.LayerNorm(d_msa)
        # 将 MSA 特征投影到隐藏维度
        self.proj_left = nn.Linear(d_msa, d_hidden)
        # 将 MSA 特征投影到隐藏维度
        self.proj_right = nn.Linear(d_msa, d_hidden)
        # 将隐藏维度的特征投影到 Pair 特征维度
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        
        # 对参数进行初始化
        self.reset_parameter()

    def reset_parameter(self):
        # 对 proj_left 和 proj_right 进行 LeCun 正态初始化
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        # 将 proj_left 和 proj_right 的偏置初始化为零
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # 将 proj_out 的权重和偏置初始化为零
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair):
        # 获取 msa 的批次大小 B、序列数量 N 和序列长度 L
        B, N, L = msa.shape[:3]
        # 对 MSA 特征进行归一化
        msa = self.norm(msa)
        # 将 MSA 特征投影到隐藏维度
        left = self.proj_left(msa)
        # 将 MSA 特征投影到隐藏维度
        right = self.proj_right(msa)
        # 对 right 特征进行归一化
        right = right / float(N)
        # 计算 left 和 right 的外积
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        # 将外积结果投影到 Pair 特征维度
        out = self.proj_out(out)
       
        # 将投影结果加到 Pair 特征上
        pair = pair + out
        
        return pair

class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        # 继承自 nn.Module，用于预测侧链扭转角
        super(SCPred, self).__init__()
        # 对查询序列的隐藏嵌入进行 Layer Normalization
        self.norm_s0 = nn.LayerNorm(d_msa)
        # 对状态特征进行 Layer Normalization
        self.norm_si = nn.LayerNorm(d_state)
        # 将查询序列的隐藏嵌入投影到隐藏维度
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        # 将状态特征投影到隐藏维度
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet 层
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # 最终输出层，用于预测扭转角
        self.linear_out = nn.Linear(d_hidden, 20)

        # 对参数进行初始化
        self.reset_parameter()

    def reset_parameter(self):
        # 对 linear_s0、linear_si 和 linear_out 进行 LeCun 正态初始化
        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        # 将 linear_s0、linear_si 和 linear_out 的偏置初始化为零
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)
        
        # 对 linear_1 和 linear_3 的权重进行 Kaiming 正态初始化
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # 将 linear_2 和 linear_4 的权重和偏置初始化为零
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq, state):
        '''
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: 对应查询序列的隐藏嵌入 (B, L, d_msa)
            - state: 前一个 SE3 层的状态特征 (B, L, d_state)
        Outputs:
            - si: 预测的扭转角 (phi, psi, omega, chi1~4 with cos/sin, Cb bend, Cb twist, CG) (B, L, 10, 2)
        '''
        # 获取 seq 的批次大小 B 和序列长度 L
        B, L = seq.shape[:2]
        # 对查询序列的隐藏嵌入进行归一化
        seq = self.norm_s0(seq)
        # 对状态特征进行归一化
        state = self.norm_si(state)
        # 将查询序列的隐藏嵌入和状态特征相加
        si = self.linear_s0(seq) + self.linear_si(state)

        # 应用 ResNet 层
        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        # 应用最终输出层
        si = self.linear_out(F.relu_(si))
        return si.view(B, L, 10, 2)


class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        # 继承自 nn.Module，用于更新结构特征
        super(Str2Str, self).__init__()
        
        # 对 MSA 特征进行 Layer Normalization
        self.norm_msa = nn.LayerNorm(d_msa)
        # 对 Pair 特征进行 Layer Normalization
        self.norm_pair = nn.LayerNorm(d_pair)
        # 对状态特征进行 Layer Normalization
        self.norm_state = nn.LayerNorm(d_state)
    
        # 将 MSA 特征和状态特征拼接后嵌入到 SE3 输入特征维度
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        # 将 Pair 特征嵌入到 SE3 边特征维度
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        # 将 SE3 边特征、rbf 特征和其他特征拼接后嵌入到 SE3 边特征维度
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        # 对 SE3 输入特征进行 Layer Normalization
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        # 对 SE3 边特征进行 Layer Normalization
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        # 对 SE3 边特征进行 Layer Normalization
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        
        # SE3 变压器模块
        self.se3 = SE3TransformerWrapper(**SE3_param)
        # 侧链预测模块
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                                   p_drop=p_drop)
        
        # 对参数进行初始化
        self.reset_parameter()

    def reset_parameter(self):
        # 对 embed_x、embed_e1 和 embed_e2 进行 LeCun 正态初始化
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # 将 embed_x、embed_e1 和 embed_e2 的偏置初始化为零
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, top_k=64, eps=1e-5):
        # 获取 msa 的批次大小 B、序列数量 N 和序列长度 L
        B, N, L = msa.shape[:3]

        if motif_mask is None:
            # 如果 motif_mask 为 None，则初始化为全零的布尔张量
            motif_mask = torch.zeros(L).bool()
        
        # 对 MSA 特征的第一个序列进行归一化
        node = self.norm_msa(msa[:,0])
        # 对 Pair 特征进行归一化
        pair = self.norm_pair(pair)
        # 对状态特征进行归一化
        state = self.norm_state(state)
       
        # 将 MSA 特征的第一个序列和状态特征拼接
        node = torch.cat((node, state), dim=-1)
        # 对拼接后的特征进行嵌入和归一化
        node = self.norm_node(self.embed_x(node))
        # 后续代码未完整展示，可能还会有对边特征的处理、SE3 模块的调用和侧链预测等操作
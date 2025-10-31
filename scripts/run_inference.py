#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""
#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle
import torch
from omegaconf import OmegaConf
import hydra
import logging
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob


def make_deterministic(seed=0):
    #  设置随机种子（对一开始输入的噪声图处理，若无约束条件（如上下文embedding等）则为随机高斯噪声图，反之则按照特殊要求）
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)#  日志初始化
    if conf.inference.deterministic:
        make_deterministic()

    #  检查可用的 GPU 并打印检查结果（有核显用核显，没核显用主显）
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    #  初始化采样器和目标/连接
    sampler = iu.sampler_selector(conf)
    
    # 添加字符输入支持
    if hasattr(conf.inference, 'character_input') and conf.inference.character_input:
        character_text = conf.inference.get('character_text', 'A')
        log.info(f"Generating protein structure for character(s): {character_text}")
        
        # 初始化字符约束采样器
        try:
            from rfdiffusion.inference.character_constraints import CharacterConstrainedSampler
            sampler = CharacterConstrainedSampler(sampler, character_text)
            log.info(f"Applied character constraints for '{character_text}'")
        except Exception as e:
            log.error(f"Failed to apply character constraints: {e}")

    #  循环多个要采样的设计
    # 初始化设计起始编号
    design_startnum = sampler.inf_conf.design_startnum
    # 当设计起始编号为-1时，自动寻找已存在的设计文件并确定新的起始编号
    if sampler.inf_conf.design_startnum == -1:
        # 查找所有符合输出前缀的现有设计文件
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        # 初始化索引列表，用于记录所有已存在的设计文件的编号
        indices = [-1]
        # 遍历所有找到的文件，提取并记录编号
        for e in existing:
            # 使用正则表达式匹配文件名中的编号
            m = re.match(".*_(\d+)\.pdb$", e)
            # 如果匹配失败，则跳过当前文件
            if not m:
                continue
            # 获取匹配到的编号
            m = m.groups()[0]
            # 将编号添加到索引列表中
            indices.append(int(m))
        # 确定新的设计起始编号为已存在编号的最大值加1
        design_startnum = max(indices) + 1

    # 遍历设计空间，生成蛋白质设计
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        # 如果配置为确定性推理模式（非随机性），则调用确定性方法
        if conf.inference.deterministic:
            make_deterministic(i_des)
        # 确定性推理模式对调试和验证模型十分重要，因为可以在确保相同的输入条件下，程序每次都产生相同的结果
        # 记录每个设计的开始时间
        start_time = time.time()
        # 构造输出文件的前缀
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        # 日志记录当前设计的前缀
        log.info(f"Making design {out_prefix}")
        # 如果谨慎模式下，设计文件已存在，则跳过当前设计（避免重复设生成相同结果，有助于防止不必要的重复计算）
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue
    
        # 初始化序列和结构
        x_init, seq_init = sampler.sample_init()
        # 初始化存储变量
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []
    
        # 将初始结构和序列复制为当前步的结构和序列
        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        # 环路反向扩散一定时间步数
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            # 执行每一步的采样
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            # 存储每一步的结果
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # 删除单例前导维
    
        # 翻转顺序以便在pymol中更好地可视化
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(
            denoised_xyz_stack,
            [
                0,
            ],
        )
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(
            px0_xyz_stack,
            [
                0,
            ],
        )
    
        # 为了日志---别乱翻
        plddt_stack = torch.stack(plddt_stack)
    
        # 保存输出结果
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]
    
        # 输出甘氨酸，基序区除外
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # 7是甘氨酸
    
        bfacts = torch.ones_like(final_seq.squeeze())
        # 对于扩散坐标，make bact=0
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        # pX0最后一步
        out = f"{out_prefix}.pdb"
    
        # 现在不要输出侧链
        writepdb(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
        )
    
        # 运行元数据
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
        )
        # 添加contig_map到元数据，如果存在的话
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        # 保存元数据到文件
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)
    
        # 如果配置为写入轨迹，则保存轨迹信息
        if sampler.inf_conf.write_trajectory:
            # 轨迹pdbs
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
    
            out = f"{traj_prefix}_Xt-1_traj.pdb"
            writepdb_multi(
                out,
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
    
            out = f"{traj_prefix}_pX0_traj.pdb"
            writepdb_multi(
                out,
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
    
        # 记录设计所花费的时间
        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()

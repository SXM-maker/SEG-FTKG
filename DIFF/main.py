import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 辅助函数 ---

def poincare_distance(u, v, eps=1e-5):
    """
    计算两个向量在庞加莱球模型中的距离。
    对应论文中的 Eq. 15。
    Args:
        u (Tensor): shape (batch_size, dim)
        v (Tensor): shape (batch_size, dim)
    Returns:
        Tensor: shape (batch_size, 1)
    """
    # 确保范数小于1，以保持在庞加莱球内
    u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True)
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)

    # 裁剪以避免数值不稳定
    u_norm_sq = torch.clamp(u_norm_sq, 0, 1 - eps)
    v_norm_sq = torch.clamp(v_norm_sq, 0, 1 - eps)

    uv_norm_sq = torch.sum((u - v) * (u - v), dim=-1, keepdim=True)

    # Eq. 15
    frac = 2 * uv_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq))
    return torch.acosh(1 + frac)


# --- GNDiff (图节点扩散模型) 模块 ---

class DenoisingNetwork(nn.Module):
    """
    一个简单的去噪网络，用于在扩散过程中预测原始数据 x0。
    论文中未指定具体架构，这里使用一个简单的MLP作为示例。
    """

    def __init__(self, dim, num_entities):
        super().__init__()
        self.num_entities = num_entities
        self.net = nn.Sequential(
            nn.Linear(dim + 1, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x_t, t_embed):
        """
        Args:
            x_t (Tensor): 带有噪声的输入, shape (batch_size, dim)
            t_embed (Tensor): 时间步t的嵌入, shape (batch_size, 1)
        """
        x = torch.cat([x_t, t_embed], dim=-1)
        return self.net(x)


class GNDiff(nn.Module):
    """
    Graph Node Diffusion (GNDiff) 模型。
    用于处理新事件（new events）。
    """

    def __init__(self, num_entities, dim, timesteps=100):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.timesteps = timesteps

        # 简单的线性噪声调度
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, axis=0)

        # 去噪网络，用于预测 x0
        self.denoising_net = DenoisingNetwork(dim, num_entities)

    def q_sample(self, x0, t):
        """
        前向过程：从 x0 添加噪声得到 xt。
        """
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)

        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1. - alpha_bar_t) * noise
        return xt, noise

    def compute_loss(self, x0):
        """
        计算扩散损失 L_diff (Eq. 8 的简化实现)。
        该实现训练网络从 xt 预测原始的 x0。
        """
        # 1. 随机选择时间步 t
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)

        # 2. 生成带噪声的 xt
        xt, _ = self.q_sample(x0, t)

        # 3. 使用网络预测 x0
        t_embed = (t.float() / self.timesteps).view(-1, 1)
        predicted_x0 = self.denoising_net(xt, t_embed)

        # 4. 计算损失（这里使用MSE，等价于简化后的L_diff）
        loss = F.mse_loss(predicted_x0, x0)
        return loss

    @torch.no_grad()
    def sample(self, s_embed, r_embed, num_samples):
        """
        反向过程：从纯噪声生成样本（新实体）。
        """
        device = s_embed.device
        x_t = torch.randn((num_samples, self.dim), device=device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device)
            t_embed = (t_tensor.float() / self.timesteps).view(-1, 1)

            # 预测 x0
            predicted_x0 = self.denoising_net(x_t, t_embed)

            # 使用 DDIM/DDPM 公式更新 x_t -> x_{t-1}
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            noise_pred = (x_t - torch.sqrt(alpha_bar_t) * predicted_x0) / torch.sqrt(1 - alpha_bar_t)

            if t == 0:
                x_t = predicted_x0
            else:
                alpha_bar_prev = self.alpha_bars[t - 1]
                noise = torch.randn_like(x_t)
                beta_t = self.betas[t]

                term1 = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
                term2 = torch.sqrt(beta_t) * noise
                x_t = term1 + term2

        return x_t


# --- DPCL (双域周期性对比学习) 模块 ---

class DPCL(nn.Module):
    """
    Dual-Domain Periodic Contrastive Learning (DPCL) 模型。
    用于处理周期性事件。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 周期性事件参数 (Eq. 16)
        self.Wp = nn.Parameter(torch.randn(dim, 2 * dim))
        self.bp = nn.Parameter(torch.randn(1, dim))

        # 非周期性事件参数 (Eq. 17)
        self.Wnp = nn.Parameter(torch.randn(dim, 2 * dim))
        self.bnp = nn.Parameter(torch.randn(1, dim))

        nn.init.xavier_uniform_(self.Wp)
        nn.init.xavier_uniform_(self.Wnp)

    def forward(self, s_embed, r_embed, o_embeds, Z_values):
        """
        计算周期性和非周期性得分。
        Args:
            s_embed (Tensor): 主体嵌入, shape (batch, dim)
            r_embed (Tensor): 关系嵌入, shape (batch, dim)
            o_embeds (Tensor): 候选客体嵌入, shape (batch, num_candidates, dim)
            Z_values (Tensor): 实体频率分析值 (Eq. 13), shape (batch, num_candidates)
        """
        s_r = torch.cat([s_embed, r_embed], dim=-1).unsqueeze(1)  # shape: (batch, 1, 2*dim)

        # --- 周期性得分 S_p (Eq. 16) ---
        # 1. 庞加莱空间处理
        # 假设 s_embed 和 o_embeds 已经被映射到庞加莱球内 (norm < 1)
        s_poincare = F.normalize(s_embed, p=2, dim=-1) * 0.99
        o_poincare = F.normalize(o_embeds, p=2, dim=-1) * 0.99
        dp = poincare_distance(s_poincare.unsqueeze(1), o_poincare)  # shape: (batch, num_candidates, 1)

        # 2. 依赖得分
        periodic_dep = torch.tanh(s_r @ self.Wp.T + self.bp)  # shape: (batch, 1, dim)
        periodic_score_base = torch.sum(periodic_dep * o_embeds, dim=-1)  # shape: (batch, num_candidates)

        S_p = periodic_score_base + Z_values + dp.squeeze(-1)

        # --- 非周期性得分 S_np (Eq. 17) ---
        # 1. 欧氏距离
        de = torch.norm(s_embed.unsqueeze(1) - o_embeds, p=2, dim=-1)  # shape: (batch, num_candidates)

        # 2. 依赖得分
        non_periodic_dep = torch.tanh(s_r @ self.Wnp.T + self.bnp)  # shape: (batch, 1, dim)
        non_periodic_score_base = torch.sum(non_periodic_dep * o_embeds, dim=-1)

        S_np = non_periodic_score_base - Z_values - de

        return S_p, S_np


# --- DPCL_Diff (主模型) ---

class DPCL_Diff(nn.Module):
    def __init__(self, num_entities, num_relations, dim, alpha=0.2, tau=0.1, gndiff_steps=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.alpha = alpha  # a in Eq. 20
        self.tau = tau  # τ for L_sup (Eq. 19)

        # 实体和关系嵌入
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)

        # 初始化子模块
        self.gndiff_module = GNDiff(num_entities, dim, timesteps=gndiff_steps)
        self.dpcl_module = DPCL(dim)

    def get_query_embedding(self, s_ids, r_ids):
        s_embed = self.entity_embeddings(s_ids)
        r_embed = self.relation_embeddings(r_ids)
        return torch.cat([s_embed, r_embed], dim=-1)

    def compute_loss(self, s_ids, r_ids, o_ids, is_new_event, Z_values, neg_o_ids):
        """
        计算总损失 L_dpcl (Eq. 20)
        Args:
            s_ids, r_ids, o_ids: (batch_size, )
            is_new_event (bool Tensor): (batch_size, ) 标记是否为新事件
            Z_values (Tensor): (batch, num_neg_samples + 1), 频率值
            neg_o_ids (Tensor): (batch, num_neg_samples), 负采样客体
        """
        s_embed = self.entity_embeddings(s_ids)
        r_embed = self.relation_embeddings(r_ids)
        o_embed = self.entity_embeddings(o_ids)

        loss_diff = torch.tensor(0.0, device=s_ids.device)
        loss_ce = torch.tensor(0.0, device=s_ids.device)
        loss_sup = torch.tensor(0.0, device=s_ids.device)

        # 1. 计算 L_diff (for new events)
        new_event_mask = is_new_event
        if new_event_mask.sum() > 0:
            # 论文中输入为 s, r, o 的拼接，这里为简化，仅对 o 进行扩散
            loss_diff = self.gndiff_module.compute_loss(o_embed[new_event_mask])

        # 2. 计算 L_ce 和 L_sup (for periodic events)
        periodic_event_mask = ~is_new_event
        if periodic_event_mask.sum() > 0:
            # 准备候选实体（正例+负例）
            all_o_ids = torch.cat([o_ids[periodic_event_mask].unsqueeze(1), neg_o_ids[periodic_event_mask]], dim=1)
            all_o_embeds = self.entity_embeddings(all_o_ids)

            # 计算得分 S_p, S_np (Eq. 16, 17)
            s_p, s_np = self.dpcl_module(
                s_embed[periodic_event_mask],
                r_embed[periodic_event_mask],
                all_o_embeds,
                Z_values[periodic_event_mask]
            )

            # 计算 L_ce (Eq. 18)
            probs_p = F.log_softmax(s_p, dim=-1)
            probs_np = F.log_softmax(s_np, dim=-1)

            # 目标是第一个实体（正例）
            target = torch.zeros(s_p.shape[0], dtype=torch.long, device=s_ids.device)
            loss_ce = -(probs_p.gather(1, target.unsqueeze(1)) + probs_np.gather(1, target.unsqueeze(1))).mean()

            # 计算 L_sup (Eq. 19)
            # 假设批次中的查询具有相同的标签（用于简化）
            query_embeds = self.get_query_embedding(s_ids[periodic_event_mask], r_ids[periodic_event_mask])
            sim_matrix = F.cosine_similarity(query_embeds.unsqueeze(1), query_embeds.unsqueeze(0), dim=-1) / self.tau

            # 同样，目标是批次内的其他样本
            sup_target = torch.arange(query_embeds.shape[0], device=s_ids.device)
            loss_sup = F.cross_entropy(sim_matrix, sup_target)

        # 3. 组合损失 (Eq. 20)
        total_loss = self.alpha * loss_diff + (1 - self.alpha) * (loss_ce + loss_sup)
        return total_loss

    @torch.no_grad()
    def predict(self, s_ids, r_ids, Z_values_all):
        """
        执行推理，结合 GNDiff 和 DPCL 的结果。
        Args:
            s_ids, r_ids: (batch_size, )
            Z_values_all: (batch_size, num_entities)
        """
        s_embed = self.entity_embeddings(s_ids)
        r_embed = self.relation_embeddings(r_ids)
        all_entities_embeds = self.entity_embeddings.weight.unsqueeze(0).repeat(s_ids.shape[0], 1, 1)

        # 1. P_diff (Eq. 21) - 从扩散模型获得概率
        # 生成的样本与所有实体计算相似度
        generated_samples = self.gndiff_module.sample(s_embed, r_embed, num_samples=s_ids.shape[0])
        diff_scores = torch.sum(generated_samples.unsqueeze(1) * all_entities_embeds, dim=-1)
        p_diff = F.softmax(diff_scores, dim=-1)

        # 2. P_dpcl (Eq. 22) - 从对比学习模型获得概率
        s_p, s_np = self.dpcl_module(s_embed, r_embed, all_entities_embeds, Z_values_all)
        total_scores = s_p + s_np  # 简单相加作为组合得分
        p_dpcl = F.softmax(total_scores, dim=-1)

        # 3. 最终概率 (Eq. 23)
        final_prob = 0.5 * (p_diff + p_dpcl)

        return final_prob


if __name__ == '__main__':
    # --- 模拟参数 ---
    NUM_ENTITIES = 1000
    NUM_RELATIONS = 50
    DIM = 128
    BATCH_SIZE = 32
    NUM_NEG_SAMPLES = 64

    # --- 实例化模型 ---
    model = DPCL_Diff(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        dim=DIM,
        alpha=0.2
    )
    print("模型 DPCL-Diff 实例化成功。")

    # --- 模拟输入数据 ---
    dummy_s = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
    dummy_r = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))
    dummy_o = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
    dummy_neg_o = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE, NUM_NEG_SAMPLES))

    # 随机一半标记为新事件
    dummy_is_new = torch.rand(BATCH_SIZE) > 0.5

    # Z_values: 模拟预计算的频率信息
    dummy_z = torch.randn(BATCH_SIZE, NUM_NEG_SAMPLES + 1)

    print("\n--- 测试损失计算 (Training) ---")
    try:
        loss = model.compute_loss(dummy_s, dummy_r, dummy_o, dummy_is_new, dummy_z, dummy_neg_o)
        print(f"损失计算成功。")
        print(f"损失值: {loss.item()}")
        assert not torch.isnan(loss), "损失值为 NaN"
        assert not torch.isinf(loss), "损失值为无穷大"
    except Exception as e:
        print(f"损失计算失败: {e}")

    print("\n--- 测试推理 (Inference/Prediction) ---")
    try:
        # 预测时需要所有实体的Z值
        dummy_z_all = torch.randn(BATCH_SIZE, NUM_ENTITIES)

        # 仅对批次中的前5个进行预测以节省时间
        predictions = model.predict(dummy_s[:5], dummy_r[:5], dummy_z_all[:5])

        print(f"推理计算成功。")
        print(f"输出概率形状: {predictions.shape} (应为: {5, NUM_ENTITIES})")
        top_k_probs, top_k_indices = torch.topk(predictions, k=5, dim=-1)
        print("每个查询的前5个预测实体ID:")
        print(top_k_indices)
        assert predictions.shape == (5, NUM_ENTITIES), "预测形状不正确"
    except Exception as e:
        print(f"推理计算失败: {e}")


import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMembership(nn.Module):
    """
    f_k(x) = exp( - (x - mu_k)^2 / (2 * sigma_k^2) )
    """

    def __init__(self, num_memberships: int, feature_dim: int):
        super().__init__()
        self.k = num_memberships
        self.feature_dim = feature_dim

        # μ_k and σ_k are learnable parameters
        self.mu = nn.Parameter(torch.randn(self.k, feature_dim))
        self.sigma = nn.Parameter(torch.ones(self.k, feature_dim))

    def forward(self, x):
        """
        x: [B, T, F]
        return: [B, T, F, K]
        """
        # 方法1：直接使用 unsqueeze 和 expand
        x = x.unsqueeze(-1)  # [B, T, F, 1]
        # x = x.expand(-1, -1, -1, self.k)  # [B, T, F, K]
        x = x.expand(-1,  -1, self.k)#[B, F, K]
        # 调整 mu 和 sigma 的形状
        # mu = self.mu.unsqueeze(0).unsqueeze(0).transpose(-1, -2)  # [1, 1, F, K]
        # sigma = self.sigma.unsqueeze(0).unsqueeze(0).transpose(-1, -2)  # [1, 1, F, K]
        mu = self.mu.unsqueeze(0).transpose(-1, -2)  # [ 1, F, K]
        sigma = self.sigma.unsqueeze(0).transpose(-1, -2)  # [1, F, K]
        gaussian = torch.exp(
            - (x - mu) ** 2 / (2 * sigma ** 2 + 1e-8)
        )
        return gaussian

        # 或者方法2：使用广播机制（更简洁）
        # x = x.unsqueeze(-1)  # [B, T, F, 1]
        # mu = self.mu.T.unsqueeze(0).unsqueeze(0)  # [1, 1, F, K]
        # sigma = self.sigma.T.unsqueeze(0).unsqueeze(0)  # [1, 1, F, K]
        #
        # gaussian = torch.exp(
        #     - (x - mu) ** 2 / (2 * sigma ** 2 + 1e-8)
        # )
        # return gaussian


class FuzzyRuleLayer(nn.Module):
    """
    S_j = sum_i w_ij * f_i(x)
    """

    def __init__(self, num_memberships: int, num_rules: int):
        super().__init__()
        self.k = num_memberships
        self.M = num_rules

        # w_ij: weight of i-th membership in j-th fuzzy set
        self.weights = nn.Parameter(
            torch.randn(self.M, self.k)
        )

    def forward(self, membership_values):
        """
        membership_values: [B, T, F, K]
        return: [B, T, F, M]
        """
        # Normalize rule weights
        weights = F.softmax(self.weights, dim=-1)  # [M, K]

        # Apply fuzzy rules
        # einsum: sum over K
        S = torch.einsum(
            'bfk,mk->bfm',
            membership_values,
            weights
        )
        return S


class Defuzzification(nn.Module):
    """
    R = (1 / M) * sum_j S_j
    """

    def __init__(self):
        super().__init__()

    def forward(self, S):
        """
        S: [B, T, F, M]
        return: [B, T, F]
        """
        return S.mean(dim=-1)


class FuzzyLogicModule(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            num_memberships: int = 5,
            num_rules: int = 2
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_memberships = num_memberships
        self.num_rules = num_rules

        self.membership = GaussianMembership(
            num_memberships=num_memberships,
            feature_dim=feature_dim
        )

        self.rules = FuzzyRuleLayer(
            num_memberships=num_memberships,
            num_rules=num_rules
        )

        self.defuzz = Defuzzification()

    def forward(self, x):
        """
        x: [B, T, F]  (center node features from GAT)
        return: [B, T, F]  (fuzzy-filtered features)
        """
        # Step 1: Membership evaluation (Eq.4)
        membership_values = self.membership(x)

        # Step 2: Fuzzy rule aggregation (Eq.5)
        S = self.rules(membership_values)

        # Step 3: Defuzzification (Eq.6)
        R = self.defuzz(S)

        return R


# 测试代码
if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    time_steps = 10
    feature_dim = 16

    model = FuzzyLogicModule(feature_dim=feature_dim)
    x = torch.randn(batch_size, time_steps, feature_dim)

    # 测试前向传播
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 检查形状是否正确
    assert output.shape == x.shape, f"Output shape {output.shape} != Input shape {x.shape}"

    # 打印中间形状
    membership_values = model.membership(x)
    S = model.rules(membership_values)
    print(f"\nMembership values shape: {membership_values.shape}")
    print(f"S shape: {S.shape}")

    print("\nModel works correctly!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
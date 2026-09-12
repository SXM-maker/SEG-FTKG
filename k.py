import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

# ====================== 1. 数据加载 ======================
file_path = 'Fig.5-1.xlsx'  # Excel文件路径
data = pd.read_excel(file_path)

# 设置绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# 提取数据
# 第一列：负样本数量 (Neg_num) - 超参数
his_lens = data.iloc[1:, 0]  # x轴：负样本数量

# 性能指标（y轴）
mrr = data.iloc[1:, 1]  # MRR (Mean Reciprocal Rank)
hit1 = data.iloc[1:, 2]  # Hits@1
hit3 = data.iloc[1:, 3]  # Hits@3
hit10 = data.iloc[1:, 4]  # Hits@10

metrics = [mrr, hit1, hit3, hit10]
metric_names = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']

# ====================== 2. 创建3D图形 ======================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 颜色映射
cmap = cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=len(metrics) - 1)

# ====================== 3. 绘制每个指标的3D曲面 ======================
for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    xs = his_lens  # x轴：负样本数量
    ys = [i] * len(xs)  # y轴：指标索引
    zs = metric  # z轴：指标值

    # 绘制3D曲线
    ax.plot(xs, ys, zs, color=cmap(norm(i)), marker='o',
            linewidth=2, markersize=6, label=name)

    # 创建填充曲面（从曲线到底部的投影）
    verts = [list(zip(xs, ys, zs)) + list(zip(xs[::-1], ys[::-1], [0] * len(zs)))]
    poly = Poly3DCollection(verts, alpha=0.3, facecolor=cmap(norm(i)))
    ax.add_collection3d(poly)

    # 标注每个数据点的具体数值
    for x, y, z in zip(xs, ys, zs):
        ax.text(x, y, z, f'{z:.1f}', fontsize=9,
                ha='center', va='bottom', alpha=0.7)

    # ====================== 4. 关键分析点标注 ======================
    # 找到性能提升最大的点（转折点）
    if len(zs) > 1:
        diffs = [zs[j] - zs[j - 1] for j in range(1, len(zs))]
        max_inc = max(diffs)
        max_inc_idx = diffs.index(max_inc) + 1

        max_x = xs[max_inc_idx]  # 最佳负样本数量
        max_y = i
        max_z = zs[max_inc_idx]

        # 用红点标记最佳转折点
        ax.scatter(max_x, max_y, max_z, color='red',
                   s=100, edgecolor='black', zorder=10)

        # 添加投影线（从最佳点到x-y平面）
        ax.plot([max_x, max_x], [max_y, max_y], [0, max_z],
                color='red', linestyle='--', linewidth=1.5, alpha=0.6)

        # 在x-y平面添加分析说明
        ax.text(max_x, max_y - 0.2, 0,
                f'Neg_num={max_x}\nOptimal',
                color='red', fontsize=10, ha='center',
                va='top', fontweight='bold', rotation=0)

# ====================== 5. 坐标轴和标签设置 ======================
# x轴：负样本数量 (Neg_num)
ax.set_xlabel("Negative Sample Number (Neg_num)", fontsize=14, labelpad=12)
ax.set_xlim(min(his_lens), max(his_lens))

# y轴：不同评价指标
ax.set_ylabel("Evaluation Metrics", fontsize=14, labelpad=12)
ax.set_yticks(range(len(metric_names)))
ax.set_yticklabels(metric_names)

# z轴：性能指标值
ax.set_zlabel("Performance Value (%)", fontsize=14, labelpad=12)
ax.set_zlim(0, max(max(mrr), max(hit1), max(hit3), max(hit10)) * 1.1)

# 设置视角
ax.view_init(elev=25, azim=130)

# ====================== 6. 添加超参数分析说明 ======================
# 添加图形标题和说明
plt.title("Sensitivity Analysis of Negative Sample Number (Neg_num)\non ICEWS05-15 Dataset",
          fontsize=16, pad=20)

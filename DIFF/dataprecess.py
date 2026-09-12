import os
import torch
import numpy as np
from collections import defaultdict


class DatasetProcessor:
    """
    处理时序知识图谱数据集，根据 DPCL-Diff 论文的实验设置。
    """

    def __init__(self, data_dir='data', dataset_name='icews14'):
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        # 用于预处理的统计信息
        self.train_entities = set()
        self.entity_frequencies = defaultdict(int)

    def load_and_map_data(self):
        """
        加载所有数据（train, valid, test）并创建实体/关系到ID的映射。
        """
        print("--- 1. 加载数据并创建映射 ---")
        all_quads = []
        for split in ['train', 'valid', 'test']:
            file_path = os.path.join(self.data_dir, self.dataset_name, f'{split}.txt')
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在，跳过。")
                continue

            with open(file_path, 'r') as f:
                for line in f:
                    s, r, o, t = line.strip().split('\t')
                    all_quads.append((s, r, o, int(t)))

        # 创建映射
        all_entities = sorted(list(set([q[0] for q in all_quads] + [q[2] for q in all_quads])))
        all_relations = sorted(list(set([q[1] for q in all_quads])))

        self.entity2id = {e: i for i, e in enumerate(all_entities)}
        self.relation2id = {r: i for i, r in enumerate(all_relations)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for i, r in self.relation2id.items()}

        print(f"实体数量: {self.num_entities}")
        print(f"关系数量: {self.num_relations}")

        return all_quads

    def split_and_process_data(self, all_quads):
        """
        将数据转换为ID，并按时间戳分割。
        论文中通常是按时间排序后，按比例或按固定时间点分割。
        这里我们假设文件已经分割好，只需加载和转换。
        """
        print("\n--- 2. 分割数据并统计训练集信息 ---")

        # 将数据转换为ID
        def quads_to_ids(quads):
            return np.array([[self.entity2id[s], self.relation2id[r], self.entity2id[o], t] for s, r, o, t in quads])

        # 加载并转换各个split
        train_quads = self._load_split('train.txt')
        valid_quads = self._load_split('valid.txt')
        test_quads = self._load_split('test.txt')

        self.train_data = quads_to_ids(train_quads)
        self.valid_data = quads_to_ids(valid_quads)
        self.test_data = quads_to_ids(test_quads)

        # 排序（非常重要）
        self.train_data = self.train_data[np.argsort(self.train_data[:, 3])]
        self.valid_data = self.valid_data[np.argsort(self.valid_data[:, 3])]
        self.test_data = self.test_data[np.argsort(self.test_data[:, 3])]

        print(f"训练集大小: {len(self.train_data)}")
        print(f"验证集大小: {len(self.valid_data)}")
        print(f"测试集大小: {len(self.test_data)}")

        # --- 关键预处理步骤 ---
        # 1. 统计训练集中出现的所有实体
        self.train_entities.update(self.train_data[:, 0])
        self.train_entities.update(self.train_data[:, 2])

        # 2. 统计训练集中每个实体的出现频率（用于计算 Z 值）
        for s, r, o, t in self.train_data:
            self.entity_frequencies[s] += 1
            self.entity_frequencies[o] += 1

        print("训练集信息统计完成。")

    def _load_split(self, filename):
        quads = []
        file_path = os.path.join(self.data_dir, self.dataset_name, filename)
        if not os.path.exists(file_path):
            return quads
        with open(file_path, 'r') as f:
            for line in f:
                s, r, o, t = line.strip().split('\t')
                quads.append((s, r, o, int(t)))
        return quads

    def get_processed_data(self, split='test'):
        """
        为指定的数据集（valid/test）生成模型所需的输入。
        这包括：
        - is_new_event: 标记事件是否为“新事件”。
        - Z_values: 计算频率分析值。
        """
        if split == 'valid':
            data = self.valid_data
        elif split == 'test':
            data = self.test_data
        else:
            raise ValueError("Split must be 'valid' or 'test'")

        print(f"\n--- 3. 为 {split} 集生成最终输入 ---")

        # 标记新事件
        # 如果一个事件的s或o在训练集中从未出现过，则为新事件。
        is_new_event = np.array([
            (s not in self.train_entities) or (o not in self.train_entities)
            for s, r, o, t in data
        ])

        # 计算 Z 值 (Eq. 13 的简化实现)
        # Z(e) = log(freq(e) + 1)。实际应用中可能更复杂。
        # 这里为每个 (s, r, o) 计算 Z(s) 和 Z(o)
        log_freq = {e_id: np.log(self.entity_frequencies.get(e_id, 0) + 1) for e_id in range(self.num_entities)}

        # Z_values_s = np.array([log_freq[s] for s, r, o, t in data])
        # Z_values_o = np.array([log_freq[o] for s, r, o, t in data])
        # Z_values_for_query = (Z_values_s + Z_values_o) / 2 # 简单平均

        # 注意：模型代码中的 Z_values 是针对候选实体的，所以它应该在生成负采样时动态计算。
        # 这里我们先准备好所有实体的频率信息。
        all_entity_log_freq = np.array([log_freq[i] for i in range(self.num_entities)])

        print(f"'{split}' 集中新事件的比例: {is_new_event.mean():.2%}")

        return {
            "quads": torch.from_numpy(data),
            "is_new_event": torch.from_numpy(is_new_event),
            "all_entity_log_freq": torch.from_numpy(all_entity_log_freq).float()
        }

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)


if __name__ == '__main__':
    # --- 1. 创建模拟数据 ---
    # 目录结构: data/dummy_dataset/train.txt, valid.txt, test.txt
    DUMMY_DATA_DIR = 'data'
    DUMMY_DATASET = 'dummy_dataset'
    os.makedirs(os.path.join(DUMMY_DATA_DIR, DUMMY_DATASET), exist_ok=True)

    # 训练集: 实体 e0-e4, 时间 1-5
    with open(os.path.join(DUMMY_DATA_DIR, DUMMY_DATASET, 'train.txt'), 'w') as f:
        f.write("e0\tr1\te1\t1\n")
        f.write("e1\tr2\te2\t2\n")
        f.write("e0\tr1\te3\t3\n")
        f.write("e2\tr3\te4\t4\n")
        f.write("e3\tr1\te1\t5\n")

    # 验证集: 包含一个已知实体(e1)和一个新实体(e5)
    with open(os.path.join(DUMMY_DATA_DIR, DUMMY_DATASET, 'valid.txt'), 'w') as f:
        f.write("e1\tr2\te3\t6\n")  # 周期性事件
        f.write("e5\tr4\te0\t7\n")  # 新事件 (e5是新的)

    # 测试集: 包含已知实体(e0)和新实体(e6)
    with open(os.path.join(DUMMY_DATA_DIR, DUMMY_DATASET, 'test.txt'), 'w') as f:
        f.write("e0\tr1\te4\t8\n")  # 周期性事件
        f.write("e4\tr5\te6\t9\n")  # 新事件 (e6是新的)

    print("模拟数据文件创建成功。")

    # --- 2. 运行数据处理器 ---
    processor = DatasetProcessor(data_dir=DUMMY_DATA_DIR, dataset_name=DUMMY_DATASET)

    # 加载和映射
    all_quads = processor.load_and_map_data()

    # 分割和统计
    processor.split_and_process_data(all_quads)

    # --- 3. 获取为模型准备的测试集数据 ---
    test_set_processed = processor.get_processed_data(split='test')

    print("\n--- 4. 检查处理结果 ---")

    # 打印实体映射
    print("实体到ID的映射:", processor.entity2id)

    # 检查训练集统计
    print("训练集实体:", processor.train_entities)
    print("实体频率:", dict(processor.entity_frequencies))

    # 检查测试集输出
    test_quads = test_set_processed['quads']
    is_new_event_flags = test_set_processed['is_new_event']
    all_freqs = test_set_processed['all_entity_log_freq']

    print("\n测试集数据:")
    for i in range(len(test_quads)):
        quad = test_quads[i].numpy()
        is_new = is_new_event_flags[i].item()
        s, r, o, t = quad[0], quad[1], quad[2], quad[3]
        s_orig = processor.id2entity[s]
        o_orig = processor.id2entity[o]

        event_type = "新事件" if is_new else "周期性事件"
        print(f"({s_orig}, r{r}, {o_orig}, t={t}) -> {event_type}")

    # 验证 is_new_event 的正确性
    # 第一个测试事件 ('e0', 'r1', 'e4', 8) -> 周期性 (e0, e4 都在训练集中)
    # 第二个测试事件 ('e4', 'r5', 'e6', 9) -> 新事件 (e6 不在训练集中)
    assert not is_new_event_flags[0].item(), "第一个测试事件应为周期性"
    assert is_new_event_flags[1].item(), "第二个测试事件应为新事件"
    print("\n断言成功，`is_new_event` 标记正确！")

    print(f"\n所有实体的Log频率值 (Z值基础) 形状: {all_freqs.shape}")

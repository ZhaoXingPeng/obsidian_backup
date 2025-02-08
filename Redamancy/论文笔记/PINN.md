# PINN (Physics-Informed Neural Networks)
## 基础概念
### PINN是什么？
PINN（Physics-Informed Neural Networks，物理信息神经网络）是一种将物理定律与深度学习相结合的创新方法。
#### 主要特点：
1. **物理约束**
   - 在神经网络的损失函数中加入物理方程作为约束条件
   - 确保模型预测符合基本物理规律
2. **混合学习方式**
   - 结合数据驱动（传统深度学习）
   - 结合物理规律（方程约束）
### 典型架构示例
```python
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
    def physics_loss(self, x, t):
        u = self.forward(x, t)
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        physics_loss = u_t - 0.01 * u_x
        return physics_loss.mean()
```

举例说明：对是否携带武器的行人的微多普勒谱图进行增强（GAN生成+物理约束）
Focante E. Physics-informed Data Augmentation for Human Radar Signatures[J]. 2023.
> *Physics-informed machine learning aims to incorporate physical prior knowledge*
> *and governing equations of the target domain into the machine learning pipeline to*
> *improve the performance of NNs in fields with limited available data but with well-*
> *defined physical models. In this thesis, physics-informed DA in the radar domain is*
> *addressed to improve the task of classifying armed and unarmed walking individuals*
> *through micro-Doppler spectrograms.*
![[Pasted image 20250208170523.png]]
![[Pasted image 20250208172711.png]]
![[Pasted image 20250208172729.png]]
[27] “Physics-informed machine learning,” Nature Reviews Physics 2021 3:6, vol. 3,
pp. 422–440, 5 2021. [Online]. Available: https://www.nature.com/articles/
s42254-021-00314-5
#### 三种方法的对比
| 方法   | 优势                       | 局限性                       | 适用场景       |
| ---- | ------------------------ | ------------------------- | ---------- |
| 观测偏置 | - 直接作用于数据层面<br>- 易于理解和实现 | - 可能计算开销大                 | 有可靠仿真模型的领域 |
| 归纳偏置 | - 保证强制执行<br>- 较好的鲁棒性     | - 难以实现复杂物理规律<br>- 架构设计难度大 | 简单几何性质和对称性 |
| 学习偏置 | - 实现灵活<br>- 易于集成         | - 约束相对松散<br>- 可能需要精细调优    | 复杂物理系统建模   |

----
# Physics-Driven Methods
- **定义**：以物理模型为主导，将机器学习作为辅助工具的方法
- **特点**：
  - 物理模型是核心
  - 机器学习用于优化或补充物理模型
  - 强调物理可解释性
- **应用场景**：
  - 传统数值方法的加速
  - 物理模型的参数优化
  - 多尺度建模
 
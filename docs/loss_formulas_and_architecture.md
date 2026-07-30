# MapAnything LiDAR + LoRA: Loss 与位姿指标说明

本文档用纯文本方式描述公式，避免 Markdown 数学块渲染问题。

参考文件：

- `scripts/train_LiDAR+LoRA.py`
- `scripts/test.py`
- `mapanything/models/mapanything/model.py`

## 1. 评测里的位姿误差怎么算

### 1.1 先做轨迹对齐

测试时先把预测轨迹整体对齐到 GT 轨迹，再算绝对误差。

做法可以简单理解为：

- 先取所有预测平移和 GT 平移
- 估计一个整体旋转 `R` 和平移 `t`
- 用同一个刚体变换去修正所有预测位姿

可写成：

`T_pred_aligned = [R, t; 0, 1] * T_pred`

这里的意思不是在代码里真的拼 LaTeX 矩阵，而是说：

- 预测轨迹先整体旋转一次
- 再整体平移一次
- 然后再和 GT 比较

### 1.2 ATE

ATE 是对齐后每帧平移的 RMSE：

`ATE = sqrt(mean(||t_pred_aligned - t_gt||^2))`

它看的是：

- 整条轨迹整体偏了多少
- 是否存在累计漂移

### 1.3 绝对旋转误差

每一帧对齐后，比较旋转矩阵：

`R_err = R_pred_aligned * R_gt^T`

再把旋转矩阵转成角度：

`rot_error = arccos((trace(R_err) - 1) / 2) * 180 / pi`

它看的是：

- 单帧姿态朝向差多少

### 1.4 绝对平移误差

每一帧对齐后直接算欧氏距离：

`abs_trans_error = ||t_pred_aligned - t_gt||_2`

它看的是：

- 单帧位置偏差

### 1.5 RPE

RPE 是相邻两帧之间的相对位姿误差。

先分别算：

`T_rel_pred = inv(T_pred_i) * T_pred_{i+1}`

`T_rel_gt = inv(T_gt_i) * T_gt_{i+1}`

然后分别算：

- 平移误差：

  `rpe_trans = ||t_rel_pred - t_rel_gt||_2`

- 旋转误差：

  `rpe_rot = angle(R_rel_pred * R_rel_gt^T)`

其中 `angle(...)` 仍然是把旋转矩阵换成角度。

RPE 的含义是：

- 不看整条轨迹是否平移偏了一点
- 重点看相邻帧运动是否一致

### 1.6 RRA / RTA

这两个不是新 loss，而是基于 RPE 的成功率指标。

- `RRA@tau`：相对旋转误差小于阈值 `tau` 的比例
- `RTA@tau`：相对平移误差小于阈值 `tau` 的比例

例如：

`RRA@1deg = fraction of frames with rpe_rot < 1 degree`

`RTA@0.1m = fraction of frames with rpe_trans < 0.1 m`

---

## 2. 训练里真正优化的位姿 loss

训练阶段不是直接优化 ATE，而是优化相邻帧的相对位姿损失。

### 2.1 平移 loss

先算相对平移差：

`delta_t = t_rel_pred - t_rel_gt`

再裁剪：

`delta_t_clipped = clip(delta_t, -1, 1)`

再做 SmoothL1：

`L_trans = SmoothL1(delta_t_clipped, 0)`

它的特点是：

- 小误差时像 L2，梯度平滑
- 大误差时像 L1，更抗异常值

### 2.2 旋转 loss

当前脚本默认用 chordal distance：

`L_rot = sqrt(2 * (3 - trace(R_pred_rel * R_gt_rel^T)) + eps)`

如果不用 chordal，就会换成角度误差：

`L_rot = arccos((trace(R_pred_rel * R_gt_rel^T) - 1) / 2)`

### 2.3 位姿总 loss

当前训练里的位姿项是：

`L_rpe = w_pose_trans * L_trans + w_pose_rot * L_rot`

---

## 3. 这个项目里其他常见 loss 的简化写法

### 3.1 深度 loss

如果开启 log 深度：

`depth_loss = mean(|log(pred_depth) - log(gt_depth)|)`

然后会去掉最大的一部分异常像素，只对剩余部分求均值。

特点：

- 对远近尺度更稳
- 不容易被极端大深度主导

### 3.2 射线方向 loss

先用内参 `K` 算出 GT 射线方向，再和预测射线做余弦距离：

`ray_loss = mean(1 - cos(pred_ray, gt_ray))`

特点：

- 直接约束“方向对不对”
- 对几何一致性很有帮助

### 3.3 相机系点云 loss

先用深度和内参把像素还原成相机系 3D 点：

`pts3d_cam_gt = depth * K^-1 * pixel_homogeneous`

然后比较预测点和 GT 点：

`pts3d_cam_loss = mean(|log(pred_pts3d_cam) - log(gt_pts3d_cam)|)`

特点：

- 约束局部三维结构
- 比单纯深度更强

### 3.4 世界系点云 loss

把相机系点云变换到世界系，再比：

`world_pts_loss = mean(|log(pred_world_pts) - log(gt_world_pts)|)`

特点：

- 约束全局空间一致性
- 对轨迹和位姿更敏感

### 3.5 confidence loss

让预测置信度去拟合“误差越小，置信度越高”的目标：

`target_conf = exp(-normalized_error)`

再做 BCE：

`confidence_loss = BCE(pred_conf, target_conf)`

特点：

- 让模型知道哪里可靠，哪里不可靠

---

## 4. 常见位姿 loss 的区别

### 4.1 L1 / L2 / SmoothL1

### L2

`||pred - gt||^2`

优点：

- 简单
- 梯度直接

缺点：

- 对异常值敏感

### L1

`||pred - gt||`

优点：

- 比 L2 更抗异常值

缺点：

- 在 0 点附近不如 L2 平滑

### SmoothL1

小误差用 L2，大误差用 L1。

优点：

- 稳定
- 常用于位姿回归

缺点：

- 需要调 beta

### 4.2 欧氏旋转损失 vs 角度损失 vs chordal

#### 欧氏旋转损失

`||R_pred - R_gt||_F^2`

优点：

- 简单

缺点：

- 不是真正的 SO(3) 距离

#### 角度损失

`angle = arccos((trace(R_pred * R_gt^T) - 1) / 2)`

优点：

- 几何意义最直接

缺点：

- 数值上在极值附近更敏感

#### chordal distance

`sqrt(2 * (3 - trace(R_pred * R_gt^T)) + eps)`

优点：

- 比角度损失更平滑
- 训练时更稳定

缺点：

- 不是严格的角度本身

### 4.3 绝对位姿 vs 相对位姿

#### 绝对位姿误差

看每一帧在全局坐标系下偏了多少。

适合：

- 看最终轨迹对不对

缺点：

- 对整体偏移比较敏感

#### 相对位姿误差

看相邻帧运动是否正确。

适合：

- 训练
- 看局部运动稳定性

优点：

- 对整体坐标系偏差更鲁棒

---

## 5. 这个项目里为什么更偏向 RPE + chordal + SmoothL1

原因很直接：

- `SmoothL1` 比纯 L2 更稳
- `chordal` 比 `acos` 角度损失更平滑
- `RPE` 比绝对位姿更适合序列训练

所以当前实现本质上是：

- 平移：`SmoothL1`
- 旋转：`chordal distance`
- 监督对象：相邻帧相对位姿

这对 LiDAR + RGB + LoRA 的联合训练更稳定。

---

## 6. 一句话总结

这个项目的位姿评测和训练可以概括为：

- 评测时：先对齐轨迹，再看 ATE / absolute rot / absolute trans / RPE
- 训练时：主要优化相邻帧的相对位姿 loss
- 旋转通常用 chordal，更稳
- 平移通常用 SmoothL1，更抗异常值


import copy
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


SRC = Path("自我介绍.pptx")
DST = Path("自我介绍_当前配置版.pptx")

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def q(prefix, tag):
    return f"{{{NS[prefix]}}}{tag}"


def get_first_run_style(tx_body):
    paragraph = tx_body.find("a:p", NS)
    if paragraph is None:
        return None, None
    p_pr = paragraph.find("a:pPr", NS)
    run = paragraph.find("a:r", NS)
    r_pr = run.find("a:rPr", NS) if run is not None else None
    return copy.deepcopy(p_pr), copy.deepcopy(r_pr)


def replace_text_body(tx_body, lines):
    if isinstance(lines, str):
        lines = [lines]
    p_pr, r_pr = get_first_run_style(tx_body)
    for child in list(tx_body):
        if child.tag == q("a", "p"):
            tx_body.remove(child)
    for line in lines:
        paragraph = ET.SubElement(tx_body, q("a", "p"))
        if p_pr is not None:
            paragraph.append(copy.deepcopy(p_pr))
        run = ET.SubElement(paragraph, q("a", "r"))
        if r_pr is not None:
            run.append(copy.deepcopy(r_pr))
        text = ET.SubElement(run, q("a", "t"))
        text.text = line


def replace_shape(shape, lines):
    tx_body = shape.find("p:txBody", NS)
    if tx_body is not None:
        replace_text_body(tx_body, lines)


def replace_cell(cell, text):
    tx_body = cell.find("a:txBody", NS)
    if tx_body is not None:
        replace_text_body(tx_body, [text])


shape_updates = {
    6: {
        2: [
            "基线模型：MapAnything，DINOv2-vitg14 视觉编码器 + 多视图几何预测头。",
            "核心问题：纯 RGB 在夜间、弱纹理和尺度模糊场景下，深度与相对位姿容易漂移。",
            "当前目标：把 LiDAR 作为几何辅助模态，为模型提供像素级深度、局部几何和绝对尺度信息。",
            "当前路线：真实相机内参投影 -> 7 通道 LiDAR 特征 -> ResNet-50 编码 -> FiLM 全局尺度调制 -> Gated Fusion + residual fusion_conv -> LoRA 联合微调。",
            "阶段性诊断：旧配置下 LiDAR=0/1 输出几乎一致，说明 LiDAR 融合链路未充分生效；当前版本重点修正投影、通道和融合激活。",
        ],
    },
    8: {
        1: ["LiDAR 7 通道生成流程（当前配置）"],
        2: ["核心预处理：使用真实相机内参，把无序点云投影成与图像对齐的 7 通道几何特征图，并额外输出全局尺度 z_d。"],
        3: [
            "内参来源：每个 seq/part 下读取 color_camera_intrinsics.txt，解析 width、height 和 K 矩阵。",
            "真实投影：假设 PCD 已在 color camera 坐标系，使用 u=fx*X/Z+cx，v=fy*Y/Z+cy；不再使用 look-at 伪前视投影。",
            "尺寸对齐：根据原始内参分辨率到训练/测试分辨率缩放 fx、fy、cx、cy。",
            "Z-Buffer：同一像素保留最近点，得到稀疏深度图、法向量图和局部几何特征图。",
            "通道组装：CAP(曲率/各向异性/平面性, 3ch) + Depth(depth/40m, 1ch) + Normal(3ch) = 7ch。",
            "LiDAR 输入统一为 9 通道：cap(3) + depth(1) + normal(3) + valid(1) + depth_edge(1)。",
            "FiLM 尺度：z_d 使用真实投影有效像素的平均 Z 深度，输入模型后取 log(z_d) 做通道级调制。",
        ],
    },
    9: {
        0: ["真实内参投影与缓存一致性"],
        1: [
            "① 解析内参：支持当前文件格式中的 width / height / K，测试脚本只从 K 块读取 9 个数，避免误把 width/height 当成 K。",
            "② 像素级投影：训练 generate_pcd_lidar_feature 与测试 get_pcd_features 使用同一套 K 缩放和投影逻辑。",
            "③ 坐标假设：当前只使用内参 K，要求 PCD 已经在 color camera 坐标系；若仍是 LiDAR 坐标系，需要补充 T_color_lidar 外参。",
            "④ 缓存版本：新增 LIDAR_CACHE_VERSION=k_projection_9ch_v1，旧 7ch 或旧投影缓存会自动失效并重算。",
            "⑤ 验证方式：分别比较 use_lidar=0/1、检查 LiDAR/fusion 参数 delta、观察 fusion gate 与 fusion_conv 是否从零附近被训练打开。",
        ],
    },
    13: {
        3: [
            "Backbone：DINOv2-vitg14，patch_size=14，输入统一 resize 到 448×448。",
            "LoRA 注入范围：encoder / info_sharing / dense_head / pose_head / scale_head 中的 Linear 层。",
            "LoRA 结构：lora_A(in,r)，lora_B(r,out)，默认 r=32、alpha=32、scaling=alpha/r。",
            "当前初始化：A 使用正态初始化，B 使用小随机值初始化，避免 B=0 导致早期梯度/更新不活跃。",
            "训练阶段：warmup 阶段应冻结 LoRA；joint 阶段启用 LoRA，与 LiDAR/Fusion 一起训练。",
            "数值安全：LoRA 分支使用 fp32 计算后再转回主干 dtype，降低混合精度下溢出风险。",
            "输出特征：RGB encoder 输出 enc_embed_dim=1536 的空间特征图，供后续 LiDAR 融合。",
        ],
    },
    14: {
        3: [
            "输入：7 通道 LiDAR 特征图，channel-last 输入后在模型内部转为 (B,C,H,W)。",
            "编码器：LiDAR encoder 的 in_chans=7；由于通道结构已改变，旧 9ch checkpoint 的首层权重不再兼容。",
            "权重策略：当前训练脚本会跳过预训练 checkpoint 中的 lidars_encoder.*，LiDAR encoder 按当前结构重新训练。",
            "FiLM：用 z_d=有效投影深度均值，模型内部计算 log(z_d)，生成 gamma/beta 对 LiDAR 特征做通道级调制。",
            "调制公式：F_lidar = F_lidar * (1 + gamma) + beta，最后一层零初始化，初始接近恒等映射。",
            "尺寸对齐：LiDAR 特征编码后会上采样到 RGB 特征图空间尺寸，再进入跨模态融合模块。",
        ],
    },
    15: {
        1: ["融合模块：对比度权重 + Gated Multimodal Fusion"],
        2: [
            "图像置信度：conf_RGB 来自灰度图 RMS 对比度，对比度越高，RGB 分支越可靠。",
            "LiDAR 权重：conf_Lidar = 0.5 / (0.5 + conf_RGB)，弱纹理或夜间场景会提高 LiDAR 相对权重。",
            "进入融合前：RGB 特征乘 conf_RGB，LiDAR 特征乘 conf_Lidar，使模态权重随图像质量动态变化。",
            "门控融合：对 RGB/LiDAR 做全局池化，拼接上下文后经 MLP 得到 gate。",
            "融合公式：F = RGB + gate * (LiDAR - RGB)，再经过 depthwise refine 卷积补充局部空间调整。",
            "初始化：gate 最后一层与 refine 零初始化，初始融合接近稳定状态，再由训练逐步打开 LiDAR 贡献。",
        ],
    },
    16: {
        1: ["融合模块：Residual fusion_conv 路径"],
        2: [
            "当前融合不是旧版“单独 1×1 Conv 替代 RGB”，而是 Gated Fusion 后叠加一条 residual fusion_conv。",
            "输入：concat([F_gated, F_lidar])，通道数为 2*enc_embed_dim，输出 enc_embed_dim。",
            "最终形式：F_out = F_gated + 0.25 * fusion_conv(concat(F_gated, F_lidar))。",
            "初始化：fusion_conv 权重和 bias 初始为 0，保证冷启动时不破坏已有 RGB 表示。",
            "训练关注点：若 fusion_conv/refine/gate 权重长期接近 0，则 LiDAR 路径可能仍未真正参与，需要通过范数、delta 和 use_lidar 消融验证。",
            "推理开关：use_lidar=True/False 控制 LiDAR 分支是否参与计算，可直接做功能性消融。",
        ],
    },
    18: {
        3: [
            "一阶段：LiDAR warmup，默认 lidar_warmup_epochs=1。目标是先让 LiDAR encoder、FiLM 和 fusion 模块产生有效信号。",
            "Warmup 可训练：lidars_encoder / lidar_film / fusion_module / fusion_conv，以及 pose/dense/scale head 基参数和 shared projection。",
            "Warmup 冻结：LoRA 参数应冻结且不进入 optimizer；RGB 权重不清零，只以 15% 概率对输入图像置零，迫使模型偶尔依赖 LiDAR。",
            "二阶段：joint training，训练 LiDAR/Fusion + LoRA；head/shared 的基参数冻结，主要通过 LoRA 适配 RGB 主干与预测头。",
            "训练开关：--lora/--no-lora 与 --lidar/--no-lidar 是功能总开关；warmup/joint 是阶段策略，由 --lidar_warmup_epochs 控制。",
            "当前诊断：warmup 切 joint 时不应重置 warmup 学到的 fusion 权重，否则会削弱第一阶段意义。",
        ],
    },
    19: {
        3: [
            "Depth Loss (w=0.1)：对有效区域的预测深度与 GT 深度做监督，约束尺度和几何形状。",
            "Ray Loss (w=0.1)：通过缩放后的真实 K 生成射线方向，约束相机几何一致性。",
            "Pts3d_cam Loss (w=0.1)：深度和 ray 相乘得到相机系点云，约束局部 3D 结构。",
            "Pose Trans Loss (w=2.0)：相邻帧平移误差，强调轨迹尺度和局部一致性。",
            "Pose Rot Loss (w=0.5)：SO(3) 旋转误差，稳定姿态估计。",
            "World/Confidence/Scale Loss：分别以 0.05 / 0.1 / 0.1 权重参与，辅助全局几何和置信度学习。",
        ],
    },
    20: {
        3: [
            "基础学习率：lr=5e-5，accum_iter=8，weight_decay=0.05。",
            "Warmup：lidar lr=1.25e-5，fusion lr=5e-6，shared lr=1.25e-5，head_base lr=2.5e-5。",
            "Joint：LiDAR lr=2.5e-5，fusion lr=1e-5，继续保持较保守的融合学习率。",
            "Head LoRA：A 使用 lr=5e-5，B 使用 lr*25=1.25e-3，利用 LoRA+ 非对称学习率加快 B 分支适配。",
            "Encoder LoRA：A 使用 lr*0.1=5e-6，B 使用 lr*25*0.1=1.25e-4，保护 DINOv2 预训练表示。",
            "梯度裁剪：LoRA / LiDAR / Fusion 分别按配置裁剪，避免某一路梯度主导训练。",
        ],
    },
    21: {
        2: [
            "LinearLR：前 warmup_steps=500 个 optimizer step 从 0.01×lr 线性升到 lr。",
            "CosineAnnealingLR：之后按 cosine 衰减，eta_min=0.01×lr。",
            "Step 时机：每个 accumulation boundary 后执行 scheduler.step()，与梯度累积保持一致。",
            "阶段切换：warmup -> joint 会重建 optimizer 和 scheduler，因此需要清空全模型梯度，避免冻结参数残留梯度污染 joint。",
        ],
    },
    22: {
        2: [
            "AMP：默认关闭，优先使用 fp32，避免 DINOv2+LoRA 在 bf16/fp16 backward 中出现系统性 NaN。",
            "NaN 防护：loss 非 finite 或梯度存在 NaN 时跳过当前 accumulation boundary 并清梯度。",
            "Gradient Clipping：在 unscale 后分别裁剪 LoRA、LiDAR、Fusion，降低训练震荡。",
            "DDP：find_unused_parameters=True，适应动态 loss 和可选 LiDAR 分支；多卡训练建议先单进程预生成 LiDAR cache。",
            "缓存一致性：当前缓存带版本号，避免旧投影/旧通道缓存被误读。",
            "诊断工具：通过 checkpoint 范数、delta、use_lidar=0/1 消融检查 LiDAR 是否真正融入。",
        ],
    },
    25: {
        0: ["LiDAR 消融诊断与当前修复"],
        2: ["实验现象：同一 checkpoint 下 use_lidar=0 与 use_lidar=1 的输出几乎一致，说明旧 LiDAR 分支虽然有参数，但功能贡献很弱。"],
        3: [
            "主要原因一：旧版使用 f=W/2 与 look-at 的伪投影，点云和 RGB 像素没有严格对齐。",
            "主要原因二：fusion_conv/refine/gate 的有效权重长期接近零，LiDAR 信号没有充分注入 RGB 表示。",
            "主要原因三：旧 9ch/缓存/checkpoint 与当前 7ch 真实投影配置不兼容，必须重新训练或只加载兼容部分。",
            "当前修复：训练和测试统一使用真实 K 投影，保留 valid/depth_edge 通道并加入缓存版本，保留 z_d 作为 FiLM 绝对尺度。",
            "验证标准：重新训练后，LiDAR=1 应在低光照/弱纹理帧上改变预测；fusion 与 LiDAR 模块 delta 应明显非零。",
        ],
        5: ["下一步重点：补充 LiDAR->camera 外参支持、避免 warmup->joint 重置 fusion、做 use_lidar 消融和分场景指标对比。"],
    },
    27: {
        2: [
            "外参支持：当前真实投影只使用 K，默认 PCD 已在 color camera 坐标系；后续需加入 T_color_lidar，覆盖原始 LiDAR 坐标系数据。",
            "融合策略：避免 joint 阶段重置 warmup 学到的 fusion，进一步确认 LoRA 在 warmup 中完全冻结/旁路。",
            "细粒度 FiLM：从全局 z_d 扩展到逐像素深度置信度或局部尺度调制。",
            "缓存与训练效率：将 LiDAR 特征预计算、分片缓存，降低 DDP 多进程小文件 I/O。",
            "评估闭环：按夜间/弱纹理/室内子场景分别做 use_lidar=0/1 消融，明确 LiDAR 对平移尺度和姿态稳定性的贡献。",
        ],
    },
}

table_updates = {
    25: [
        "诊断项", "旧现象", "当前处理", "验证方式",
        "投影", "伪前视 f=W/2", "真实 K 像素投影", "可视化投影覆盖 RGB",
        "通道", "历史配置混杂", "固定 7ch", "检查 pcd shape",
        "缓存", "旧缓存可能复用", "cache version", "miss 后重算",
        "融合", "权重接近 0", "Gated + residual", "范数/delta/消融",
        "尺度", "只看参数存在", "z_d -> FiLM", "检查 scale 分布",
    ]
}


def main():
    with zipfile.ZipFile(SRC, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    for slide_no, updates in shape_updates.items():
        name = f"ppt/slides/slide{slide_no}.xml"
        root = ET.fromstring(files[name])
        shapes = root.findall(".//p:sp", NS)
        for idx, lines in updates.items():
            if idx >= len(shapes):
                raise IndexError(f"slide {slide_no} shape {idx} not found")
            replace_shape(shapes[idx], lines)
        if slide_no in table_updates:
            frames = root.findall(".//p:graphicFrame", NS)
            if frames:
                cells = frames[0].findall(".//a:tc", NS)
                values = table_updates[slide_no]
                for i, cell in enumerate(cells):
                    replace_cell(cell, values[i] if i < len(values) else "")
        files[name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(DST, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)

    print(DST)


if __name__ == "__main__":
    main()

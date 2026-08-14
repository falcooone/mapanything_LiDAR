from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


@dataclass
class Section:
    title: str
    paragraphs: Sequence[str]


def _paragraph_xml(text: str, style: Optional[str] = None) -> str:
    lines = text.split("\n")
    runs = []
    for i, line in enumerate(lines):
        if i > 0:
            runs.append("<w:r><w:br/></w:r>")
        if line == "":
            runs.append("<w:r><w:t xml:space=\"preserve\"> </w:t></w:r>")
        else:
            runs.append(f"<w:r><w:t xml:space=\"preserve\">{escape(line)}</w:t></w:r>")
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def _page_break_xml() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def _build_document_xml(paragraphs: Sequence[str]) -> str:
    body = "".join(paragraphs)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
      <w:cols w:space="708"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def _build_styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="120"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
      <w:sz w:val="22"/>
      <w:szCs w:val="22"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="240"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="32"/>
      <w:szCs w:val="32"/>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="200" w:after="80"/></w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="160" w:after="60"/></w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="22"/>
      <w:szCs w:val="22"/>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListBullet">
    <w:name w:val="List Bullet"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:ind w:left="360" w:hanging="360"/>
      <w:spacing w:after="60"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def _build_content_types_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{CONTENT_TYPES_NS}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""


def _build_root_rels_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def _build_document_rels_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}"/>
"""


def _write_docx(path: Path, paragraphs: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _build_content_types_xml())
        zf.writestr("_rels/.rels", _build_root_rels_xml())
        zf.writestr("word/document.xml", _build_document_xml(paragraphs))
        zf.writestr("word/styles.xml", _build_styles_xml())
        zf.writestr("word/_rels/document.xml.rels", _build_document_rels_xml())


def _build_doc(title: str, sections: Sequence[Section]) -> List[str]:
    paragraphs: List[str] = []
    paragraphs.append(_paragraph_xml(title, "Title"))
    paragraphs.append(_paragraph_xml("生成说明：本文件由仓库内论文 PDF 和 MapAnything 源码整理而成。", "Normal"))
    paragraphs.append(_paragraph_xml("阅读顺序建议：先看“论文定位”，再看“方法结构”，最后看“与 MapAnything fusion 的关系”。", "Normal"))
    for section in sections:
        paragraphs.append(_paragraph_xml(section.title, "Heading1"))
        for item in section.paragraphs:
            if item.startswith("• "):
                paragraphs.append(_paragraph_xml(item[2:], "ListBullet"))
            elif item.startswith("  ") and not item.strip().startswith("—"):
                paragraphs.append(_paragraph_xml(item.strip(), "Heading3"))
            else:
                paragraphs.append(_paragraph_xml(item, "Normal"))
    return paragraphs


MAST3R_SECTIONS = [
    Section(
        "1. 论文定位",
        [
            "MASt3R 的核心目标不是单纯做匹配，而是把“匹配”重新定义成一个 3D 问题：先预测两张图各自的 3D pointmap，再在 3D/特征空间中找对应关系。",
            "它是在 DUSt3R 基础上做的增强版本。DUSt3R 已经能从两张未标定图像里回归 3D 几何，但其对应关系更多是副产品，精度不够高；MASt3R 的改进就是增加一个专门的 dense local feature head，并用匹配损失直接训练它。",
            "这篇论文的目标非常明确：在保持 DUSt3R 对大视角变化鲁棒性的前提下，把匹配精度、定位效果和高分辨率可用性都抬上去。",
        ],
    ),
    Section(
        "2. 核心直觉",
        [
            "作者认为，传统匹配把问题当成 2D keypoint/descriptor 问题，会丢失全局几何上下文；而真实的对应点本质上是同一个 3D 点的两个投影，所以应在 3D 语义上建模。",
            "MASt3R 保留 DUSt3R 的 pointmap 回归主干，但额外学习 dense feature maps，让网络可以直接输出更适合做最近邻匹配的表示。",
            "因此，MASt3R 的定位可以概括为：它是一个“面向匹配的 3D 重建网络”，不是一个只输出 matches 的纯匹配器。",
        ],
    ),
    Section(
        "3. 方法结构",
        [
            "整个结构仍然是双图像、Siamese 编码、双解码器的框架，但在解码后的表示上增加两种预测头：3D 回归头和 descriptor 头。",
            "输入两张图像后，编码器分别提取表示，再通过 intertwined decoders 交换信息，理解跨视角的空间关系。最终输出包括 pointmap 和 confidence map，以及专门给匹配使用的 local features。",
            "文中的关键点是：MASt3R 不是把匹配塞到后处理里，而是让网络在训练时就知道“我输出的特征就是为了匹配”。",
        ],
    ),
    Section(
        "4. 损失函数",
        [
            "3D 回归部分继续沿用 DUSt3R 的监督：让预测 pointmap 接近 GT pointmap，并结合置信度项进行加权。",
            "匹配部分使用 InfoNCE 风格的对比损失。对于一组已知真值对应点，网络必须把正确点的特征拉近，并把其他位置推远，这比“回归到附近就算对”更严格，因此能显著提升精度。",
            "总损失可以理解为：L_total = L_conf + λ * L_match。前者约束几何，后者约束可匹配性。",
        ],
    ),
    Section(
        "5. 快速 reciprocal matching",
        [
            "如果对所有像素做全量两两最近邻，复杂度是二次的，代价太高。MASt3R 的关键工程贡献之一就是 fast reciprocal matching。",
            "它不是直接枚举所有像素对，而是从一个稀疏初始集合出发，迭代地做最近邻搜索、找 reciprocal pairs、删除已收敛点，逐步收敛到稳定匹配。",
            "这样做的好处有两个：一是速度快很多，二是由于 reciprocal 约束，往往会过滤掉一些不稳定对应，最后反而比全量暴力匹配更稳。",
        ],
    ),
    Section(
        "6. coarse-to-fine 高分辨率策略",
        [
            "MASt3R 原始训练分辨率通常是 512 像素最大边，而高分辨率图像如果直接缩小，会损失局部细节。",
            "论文的做法是先在缩小版本上得到粗对应，再在多个重叠 crop 上逐步细化，并把匹配结果映射回原分辨率。",
            "这使得 MASt3R 能在大图上使用多尺度匹配，而不是被训练分辨率锁死。",
        ],
    ),
    Section(
        "7. 实验结论",
        [
            "论文最重要的结论是：MASt3R 在 Map-free localization、relative pose estimation、multi-view 3D reconstruction 等任务上，整体优于 DUSt3R 和此前的匹配方法。",
            "尤其在极端视角变化、重复纹理、低纹理场景中，它仍然能输出较稳的 dense correspondences，这也是作者强调“3D-aware matching”的原因。",
            "代价是它仍然是两视图主导的框架，并且需要 post-processing/global alignment 来扩展到更多视图。",
        ],
    ),
    Section(
        "8. 和 MapAnything fusion 的关系",
        [
            "• MASt3R 更像一个“高质量双图几何基础模块”，适合作为匹配/重建的 pairwise backbone。",
            "• MapAnything fusion 是系统级框架，强调把多视图图像、几何先验和 LiDAR 等输入统一融合到同一流水线里。",
            "• 在本仓库中，MASt3R 通过 `mapanything/models/external/mast3r/__init__.py` 接入，输出被统一成 MapAnything 风格的 `pts3d / cam_trans / cam_quats`。",
            "• 结构上，MASt3R 的强项是匹配精度与鲁棒性；MapAnything 的强项是“多模态融合 + 多任务统一输出”。",
        ],
    ),
    Section(
        "9. 一句话总结",
        [
            "如果把 MASt3R 压缩成一句话：它是在 DUSt3R 的 3D pointmap 之上，额外训练了可匹配的 dense feature，并用更高效的 reciprocal matching 把它真正变成一个强匹配器。",
        ],
    ),
]


VGGT_SECTIONS = [
    Section(
        "1. 论文定位",
        [
            "VGGT 的目标比 MASt3R 更“通用”：它要在一个 feed-forward 网络里同时预测相机参数、depth map、point map 和 point tracks。",
            "它强调的是“直接从多张图像里一次前向得到完整 3D 属性”，尽量不依赖后处理优化。作者的立场很明确：3D 任务可以像大语言模型一样，由一个大模型统一学出来。",
            "因此，VGGT 不是某个单点任务的优化器，而是一个 3D foundation model。",
        ],
    ),
    Section(
        "2. 输入输出定义",
        [
            "输入是一个图片序列，可以是 1 张、几张，甚至上百张图。输出则是一组按帧对应的 3D 属性：camera parameters、depth maps、point maps、tracking features 和 point tracks。",
            "相机参数采用 quaternions + translation + field of view 的表示。点图和深度图都以第一帧为世界参考系，体现了它的“reference frame”设计。",
            "轨迹不是直接由主干输出，而是由特征再喂给单独的 tracking module 生成，这样做把几何表示和跟踪解耦。",
        ],
    ),
    Section(
        "3. 架构主干",
        [
            "VGGT 的主干是一个大 transformer，关键设计是 alternating attention：在 frame-wise self-attention 和 global self-attention 之间交替。",
            "这样做的原因很直接：既要让每帧内部先充分建模，又要让不同帧之间共享信息。",
            "它的 inductive bias 很少，核心只有两点：使用 DINO 风格 patch token，以及在 frame/global 两个粒度间切换。",
        ],
    ),
    Section(
        "4. Camera token 与 register token",
        [
            "每个输入图像都会附加 camera token 和 register tokens。第一帧和其他帧使用不同的 learnable token，这样模型能知道“谁是参考帧”。",
            "第一帧的相机外参被固定成 identity，这使得所有几何预测都天然落在第一相机坐标系下。",
            "这和很多传统多视图方法需要显式选 world frame 不同，VGGT 直接把参考系写进 token 设计里。",
        ],
    ),
    Section(
        "5. 预测头",
        [
            "Camera head 从 camera tokens 预测 camera parameters。",
            "DPT head 从 image tokens 预测 dense outputs，包括 depth maps、point maps 和 tracking features。",
            "Tracking head 再基于 dense tracking features 和 query point 生成整条轨迹。也就是说，VGGT 的 tracking 是一个上层模块，不是简单的特征相关就结束。",
        ],
    ),
    Section(
        "6. 训练思想",
        [
            "VGGT 的一个重要观点是：即使 depth、camera、point map 之间有确定性关系，也应该把这些量都显式纳入训练，因为多任务联合监督会带来更强的表示学习。",
            "不过在推理时，它又指出可以从 depth + camera 重新导出更好的 point map，这说明训练和推理的最优使用方式并不完全相同。",
            "这是一种典型的“训练时多头监督，推理时择优组合”的思路。",
        ],
    ),
    Section(
        "7. 实验结论",
        [
            "VGGT 在相机姿态估计、multi-view depth estimation、point map estimation、image matching 和 tracking 上都给出强结果。",
            "论文最显著的卖点是：它通常比依赖 BA / global alignment 的方法更快，而且很多时候精度还更高。",
            "如果再加 BA，性能还能进一步涨，这意味着 VGGT 自身已经给出了很好的初始化。",
        ],
    ),
    Section(
        "8. 和 MapAnything fusion 的关系",
        [
            "• 在本仓库里，VGGT 通过 `mapanything/models/external/vggt/__init__.py` 接入，`configs/model/vggt.yaml` 里将输入归一化设为 `identity`。",
            "• 这说明 MapAnything 不是重新实现 VGGT，而是把它当成一个高质量的外部几何 backbone 来统一调用。",
            "• VGGT 更像“统一 3D 基础模型”；MapAnything fusion 更像“把这种基础模型与其他几何/传感器信号做系统级融合”。",
            "• 与 MapAnything 不同，VGGT 本身并不以 LiDAR 融合为核心，也不是一个显式的多模态 prior injection 模块。",
        ],
    ),
    Section(
        "9. 一句话总结",
        [
            "VGGT 的本质是：用一个几乎不带特定 3D 先验的大 transformer，把多视图几何、相机、深度和轨迹一起学出来，并且在很多任务上直接替代传统几何优化。",
        ],
    ),
]


POW3R_SECTIONS = [
    Section(
        "1. 论文定位",
        [
            "Pow3R 的核心不是“更大”，而是“更可控”。它要解决的是：当测试时已经有额外先验时，如何把这些先验塞进一个现成的 3D 重建网络里，并且从中获益。",
            "它建立在 DUSt3R 范式之上，但和 MASt3R 不同，Pow3R 不强调匹配头，而是强调可选输入条件：camera intrinsics、relative pose、sparse/dense depth 等。",
            "作者的主张可以概括为：真实世界里往往不只提供 RGB，所以 3D 模型应当接受任意 subset 的几何先验。",
        ],
    ),
    Section(
        "2. 输入空间",
        [
            "Pow3R 可以在训练/测试时接受不同子集的辅助信息：K1/K2、D1/D2、相对位姿 P12，必要时还能从稀疏深度或点云中获益。",
            "这点和 DUSt3R / MASt3R 非常不同。后两者主要是 RGB-only，而 Pow3R 明确把 priors 设计成第一等公民。",
            "这也让它可以做一些原本不容易做的事情，比如 native resolution 的滑窗推理和 depth completion。",
        ],
    ),
    Section(
        "3. 架构主干",
        [
            "它仍然沿用 DUSt3R 的 pointmap 逻辑：两张图先编码，再通过双解码器回归 pointmaps。",
            "与 DUSt3R 的关键差异在于：Pow3R 额外预测第二张图在自身坐标系中的 pointmap X2,2，从而能更直接地恢复相对姿态。",
            "从实现角度看，Pow3R 本质上是“DUSt3R + 条件化输入 + 第二个 pointmap 目标 + 更灵活的推理入口”。",
        ],
    ),
    Section(
        "4. 如何注入先验",
        [
            "论文给出两种主策略：embed 和 inject-n。",
            "embed 是把辅助信息先编码成 token embedding，再在第一层前加到输入 token 上。",
            "inject-n 则是在若干 transformer block 内插入专门的模块，把先验直接注入到中间层。",
            "实验显示，inject-1 往往是一个很好的折中：比纯 embed 更强，但比在很多层里都注入更轻。",
        ],
    ),
    Section(
        "5. 任务层面的输出",
        [
            "点图的 z 轴可以直接得到 depth map。",
            "相机焦距可以由 pointmap 反求；相对位姿可以由两帧 pointmap 做 Procrustes alignment 得到。",
            "高分辨率推理时，Pow3R 可以借助 crop intrinsics 做 sliding-window inference，再把窗口结果拼回去。",
        ],
    ),
    Section(
        "6. 训练方式",
        [
            "训练时，模型会随机采样不同的 modality subset，这一点非常关键。",
            "这样做不是为了“缺省鲁棒”，而是为了让模型真的学会处理各种 priors 的组合，而不是只会一种固定输入模式。",
            "作者还会做 aggressive non-centered cropping，让模型在训练时就适应偏心 crop 和高分辨率场景。",
        ],
    ),
    Section(
        "7. 主要实验结论",
        [
            "当不给任何辅助信息时，Pow3R 大致能接近 DUSt3R；但当给出 intrinsics、depth 或 pose 时，性能会明显提升。",
            "它在 depth completion、multi-view depth estimation、pose estimation 和高分辨率滑窗场景中表现很强。",
            "论文强调它是一个能“吃进任意 subset priors”的模型，这一点是它区别于现有 RGB-only 3D 模型的核心。",
        ],
    ),
    Section(
        "8. 和 MapAnything fusion 的关系",
        [
            "• Pow3R 跟 MapAnything fusion 的理念最接近：两者都重视把额外几何先验作为输入，而不是只吃 RGB。",
            "• 但 Pow3R 仍是双图像、条件化 pointmap regression 框架；MapAnything fusion 是更通用的多视图、多模态统一框架。",
            "• 在本仓库里，Pow3R 通过 `mapanything/models/external/pow3r/__init__.py` 接入，`configs/model/pow3r.yaml` 仍然标记 `dust3r` 归一化，这说明它在接口上还是 DUSt3R 家族。",
            "• 如果你把 MapAnything 看成系统层，Pow3R 更像一个“可注入 priors 的几何底座”；MapAnything 则把这种底座进一步扩大为多传感器融合框架。",
        ],
    ),
    Section(
        "9. 一句话总结",
        [
            "Pow3R 的本质是：在 DUSt3R 的 pointmap 世界里，把 camera intrinsics、relative pose 和 depth priors 变成可训练、可推理的条件输入，从而让模型更像一个真正可控的 3D 几何求解器。",
        ],
    ),
]


COMPARE_SECTIONS = [
    Section(
        "1. 总体结论",
        [
            "这三篇论文和 MapAnything fusion 的关系，可以用一句话概括：MASt3R 解决“更准的匹配”，VGGT 解决“更统一的 3D 基础模型”，Pow3R 解决“如何把测试时可获得的先验真正用起来”；MapAnything fusion 则把这些思想进一步系统化，做成多模态、多视图、可插拔的统一框架。",
        ],
    ),
    Section(
        "2. 任务层面对比",
        [
            "• MASt3R 的主任务是 3D-aware matching，兼顾重建。",
            "• VGGT 的主任务是多视图全局 3D 属性回归，强调单次前向完成相机、深度、point map 和跟踪。",
            "• Pow3R 的主任务是 prior-conditioned 3D reconstruction，强调任意 subset 条件输入。",
            "• MapAnything fusion 的主任务是把多种几何/传感器/外部模型统一到同一个 pipeline 里，并输出统一格式的几何结果。",
        ],
    ),
    Section(
        "3. 输入设计对比",
        [
            "MASt3R：主要是 RGB pair，几乎不依赖额外先验。",
            "VGGT：多图像输入，几乎不依赖显式几何 priors，靠大规模训练自己学。",
            "Pow3R：RGB + 任意子集的 intrinsics / pose / depth priors。",
            "MapAnything fusion：RGB 之外还显式支持 ray directions、depth、camera pose、LiDAR 等，并且把不同模态分别编码后再融合。",
        ],
    ),
    Section(
        "4. 结构对比",
        [
            "MASt3R / Pow3R / VGGT 都是“单模型端到端”思路，只是任务重点不同。",
            "MapAnything fusion 更像“系统架构”：先分别编码，再通过 `GatedMultimodalFusion`、通道残差和 multi-view transformer 做统一信息共享。",
            "在代码上，MapAnything 的关键实现位于 `mapanything/models/mapanything/model.py`，而外部模型只是通过 wrapper 接进来。",
        ],
    ),
    Section(
        "5. MapAnything 的融合特点",
        [
            "MapAnything 对 LiDAR 的处理不是简单拼接，而是：先单独编码 LiDAR，再用 reliability map、FiLM 风格的全局尺度调制、门控融合和残差卷积把它并入图像特征。",
            "这和 Pow3R 的“把先验加到 transformer 输入 token 上”不同，也和 VGGT 的“几乎不注入先验”不同。",
            "MapAnything 更强调多模态之间的可控融合，而不是把所有东西都塞进同一个黑盒 transformer。",
        ],
    ),
    Section(
        "6. 在本仓库中的接入方式",
        [
            "MASt3R、VGGT、Pow3R 都通过 `mapanything/models/external/.../__init__.py` 作为外部模型接入。",
            "对应配置文件分别是 `configs/model/mast3r.yaml`、`configs/model/vggt.yaml`、`configs/model/pow3r.yaml`。",
            "MapAnything 自身则由 `configs/model/mapanything.yaml` 管理，说明它既能作为独立模型，也能把这些外部几何模型当作可插拔部件。",
        ],
    ),
    Section(
        "7. 选择建议",
        [
            "如果你要的是高质量对应点或匹配，优先理解 MASt3R。",
            "如果你要的是一个能直接输出相机、深度、点图、轨迹的统一 3D 模型，优先理解 VGGT。",
            "如果你有 intrinsics / depth / pose 这类先验并想把它们用掉，优先理解 Pow3R。",
            "如果你要做多模态 LiDAR + RGB 融合并希望兼容多种外部几何模型，优先理解 MapAnything fusion。",
        ],
    ),
    Section(
        "8. 与 MapAnything fusion 的本质差异",
        [
            "MASt3R / VGGT / Pow3R 更像“模型论文”；MapAnything fusion 更像“平台论文 / 系统论文”。",
            "前者关心某一类几何建模是否更强；后者关心如何把这些不同范式的几何模型和多传感器输入统一起来。",
            "如果把前者比作发动机，MapAnything fusion 更像整车控制系统。",
        ],
    ),
]


def build_all_docs(output_dir: Path) -> List[Path]:
    docs = [
        ("MASt3R_精读笔记.docx", "MASt3R 精读笔记", MAST3R_SECTIONS),
        ("VGGT_精读笔记.docx", "VGGT 精读笔记", VGGT_SECTIONS),
        ("Pow3R_精读笔记.docx", "Pow3R 精读笔记", POW3R_SECTIONS),
        ("MapAnything_vs_三者对比.docx", "MapAnything fusion 对比笔记", COMPARE_SECTIONS),
    ]
    written = []
    for filename, title, sections in docs:
        paragraphs = _build_doc(title, sections)
        out_path = output_dir / filename
        _write_docx(out_path, paragraphs)
        written.append(out_path)
    return written


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output" / "reading_notes"
    written = build_all_docs(output_dir)
    manifest = output_dir / "README.txt"
    manifest.write_text(
        "\n".join(str(p) for p in written) + "\n",
        encoding="utf-8",
    )
    print("Generated:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

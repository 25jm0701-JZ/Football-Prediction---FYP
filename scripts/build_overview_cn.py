#!/usr/bin/env python3
"""生成项目总览中文版 docx。"""
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "Project_Overview_CN.docx"
doc = Document()

NAVY = RGBColor.from_string("1F4E79")
BLACK = RGBColor(0, 0, 0)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = NAVY

def add_body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)

def add_bullet(text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)

# ═════════════════════════════════════════════════════════
add_heading("足球赛果预测 —— 项目总览", 1)
add_body("Final Year Project — Group 26")
add_body("基于逻辑回归与滚动统计特征的英格兰职业足球赛果预测系统，附基于 Buchdahl (2003) 框架的价值投注改进。")

# ═════════════════════════════════════════════════════════
add_heading("一、双轨结构", 1)

add_heading("Track A：英格兰四级联赛主模型", 2)
add_bullet("研究问题：仅用免费公开比赛数据（无任何付费 API），统计模型能否稳定优于博彩市场？")
add_bullet("数据：英格兰四级联赛（E0-E3）2019-2025 共 6 赛季，11,952 场")
add_bullet("特征：62 维——滚动球队状态（5/10 场窗口）、射门数据、历史交锋、联赛等级标识。不含任何赔率特征。")
add_bullet("模型：Logistic 回归（L2 正则化，lbfgs，C=1.0）")
add_bullet("核心结果（walk-forward）：57.25% vs 市场 49.40%，领先 +7.85 个百分点")
add_bullet("价值投注（30% 阈值，walk-forward）：7,782 注，ROI +55.0%")
add_bullet("参考论文：特征工程方法借鉴 Atta Mills et al. (2024)；LR 经本实验验证为最优（稳定性+校准）；价值投注框架参考 Buchdahl (2003)")

add_heading("Track B：FootyStats 增强模型（价值投注改进）", 2)
add_bullet("研究问题：跨赛季 PPG 特征能否改进 Buchdahl (2003) 的评分系统，提升投注回报？")
add_bullet("数据：英超 2019-2025 共 6 赛季，2,280 场，71 维特征（62 基础 + 9 FootyStats）")
add_bullet("核心结果：68.42%（+6.14pp 提升自滚动统计基线）")
add_bullet("Buchdahl 对照：原文 ROI +2~10% → 本论文 +5~21%")
add_bullet("局限：FootyStats 仅限英超。若低级别联赛有类似数据，预计可缩小约 9.5pp 性能差距")

# ═════════════════════════════════════════════════════════
add_heading("二、价值投注方法论", 1)

add_heading("2.1 Edge（优势）", 2)
add_body("Edge = P_model / P_market - 1")
add_body("P_model 为模型预测概率，P_market 为 Bet365 收盘赔率经去抽水后的市场概率。Edge > 0 表示模型认为市场低估了该结果。")

add_body("案例——利物浦 vs 水晶宫（2023-24 赛季）：")
add_bullet("模型预测客胜概率：28.1%")
add_bullet("市场隐含概率：8.6%（对应赔率 11.0）")
add_bullet("Edge = 0.281 / 0.086 - 1 = 227%")
add_bullet("实际结果：水晶宫客胜 ✅。1 单位投注获 10 单位利润。")

add_heading("2.2 Threshold（阈值）", 2)
add_body("阈值是下注的最低 Edge 门槛。只投注 Edge 超过该值的结果。阈值越高→投注数越少→单注平均质量越高。")

add_body("模型评估与阈值扫描使用两种不同的验证方法：")

add_body("模型评估采用 Walk-forward 逐季滚动验证——每赛季用之前所有赛季的数据训练，预测下一整个赛季。跨 5 轮（2020-2025），共 10,180 场测试，模拟真实部署场景。模型加权平均准确率 57.46%，市场 49.40%，平均领先 **+8.06pp**。")

add_body("以下为 walk-forward 全阈值扫描结果（每场比赛只投注 Edge 最高的一个 outcome，即 best per match 策略）：")

add_bullet("阈值 0%：10,180 注，ROI +41.8%（每场必投）")
add_bullet("阈值 30%：7,782 注，ROI +55.0%（论文推荐）")
add_bullet("阈值 50%：5,658 注，ROI +66.0%")
add_bullet("阈值 100%：2,441 注，ROI +99.3%")
add_bullet("阈值越高 ROI 越高，但注数减少、统计可靠性下降。论文以 30% 阈值作为主结果，平衡注数与 ROI。")

add_heading("2.3 投注策略", 2)
add_body("本文采用 Best per match 策略——每场比赛从主胜、平局、客胜三个结果中选出 Edge（优势）最高的那个进行投注，若该 Edge 超过阈值则下注 1 单位。一场最多一注。这是最直接且样本量最大的部署方式。")

add_heading("2.4 市场赔率处理", 2)
add_body("原始 Bet365 收盘赔率 → 取倒数（1/赔率）→ 去抽水（归一化到总和=1）→ 市场概率。")
add_body("示例：利物浦 vs 水晶宫，庄家赔率主胜 1.2、平 7.5、客 11.0。去抽水后市场概率分别为 78.8%、12.6%、8.6%。")

add_heading("2.5 与 Buchdahl (2003) 的对比", 2)
add_body("Buchdahl 原文方法：计算近 6 场进球差评分 → 查历史概率表（14,002 场统计）→ 算公平赔率 → 与庄家比较 → 投主胜。")
add_body("本论文改进：LR 模型从 71 维特征直接输出概率 → Edge = P_model/P_market - 1 → 与阈值比较 → 可投任意结果。")

# ═════════════════════════════════════════════════════════
add_heading("三、实验结果（Track A）", 1)

add_heading("3.1 验证方法：Walk-forward", 2)
add_body("本文使用 Walk-forward 逐季滚动验证——用已完成的赛季训练模型，预测下一个完整赛季。然后扩增训练集（包含刚预测完的赛季），再预测下一季。重复 5 轮，累计 10,180 场测试。这种方法模拟真实部署场景，排除了单一年份的偶然性，跨 5 个独立年份验证模型稳定性。")
add_body("每场比赛的特征只使用该场比赛之前的数据（shift(1)），确保没有任何未来信息泄露。")

add_heading("3.2 Track A：模型准确率", 2)
add_body("Walk-forward（5 折，10,180 场测试）：")

table = doc.add_table(rows=5, cols=4)
table.style = 'Light Grid Accent 1'
headers = ['指标', '模型', '市场', '差值']
data = [
    ['准确率', '57.25%', '49.40%', '+7.85pp'],
    ['LogLoss', '0.9003', '1.0180', '-0.1177'],
    ['Brier', '0.1778', '0.2034', '-0.0256'],
    ['ROC AUC', '0.7331', '0.6302', '+0.1029'],
]
for j, h in enumerate(headers):
    table.rows[0].cells[j].text = h
for i, row in enumerate(data):
    for j, val in enumerate(row):
        table.rows[i+1].cells[j].text = val

add_body("按类别：客胜 70.3%（市场 47.4%）、平局 13.0%（市场 0.0%）、主胜 81.8%（市场 80.2%）。没有一年模型低于市场超过 2pp。价值投注（30% 阈值）：7,782 注，ROI +55.0%。")

add_heading("3.3 Track B：FootyStats 增强", 2)
add_bullet("滚动特征基线：62.28%")
add_bullet("+FootyStats：68.42%（+6.14pp）")
add_bullet("市场赔率：57.02%")

add_heading("3.4 模型 vs 市场分歧分析", 2)
add_bullet("测试集：1,900 场英超（walk-forward），LR + FootyStats 模型")
add_bullet("模型与市场在 38.6% 的比赛上意见不一致")
add_bullet("分歧时模型正确率 45.8%，市场正确率 25.6%，模型/市场比 **1.8 倍**")
add_body("案例一——利物浦 vs 水晶宫（2023-24）：市场看主胜概率 78.8%，模型给水晶宫客胜 28.1%（市场仅 8.6%），Edge 227%。实际水晶宫 1-0 客胜。")
add_body("案例二——伯恩茅斯 vs 阿森纳（2024-25）：市场看主胜概率 33%，模型看客胜概率 98%（Edge 200%）。实际阿森纳客胜。模型在强队客场作战时能识别市场对主场优势的高估。")

# ═════════════════════════════════════════════════════════
add_heading("四、与文献对比", 1)

table2 = doc.add_table(rows=4, cols=4)
table2.style = 'Light Grid Accent 1'
h2 = ['维度', 'Buchdahl (2003)', 'Reade et al. (2020)', '本论文']
d2 = [
    ['数据', '英联赛 1993-2001', 'PL 2 赛季', 'E0-E3 6 赛季 11,952 场'],
    ['模型', '进球差评分', '基本泊松', 'LR + 62~71 维'],
    ['分类 vs 市场', '未报告', '模型≈市场', '+8.90pp'],
]
for j, h in enumerate(h2):
    table2.rows[0].cells[j].text = h
for i, row in enumerate(d2):
    for j, val in enumerate(row):
        table2.rows[i+1].cells[j].text = val

# ═════════════════════════════════════════════════════════
add_heading("五、局限与未来方向", 1)
add_bullet("FootyStats 仅覆盖英超 6 季。PPG/xG/控球率贡献了 +6.14pp 提升，但低级别联赛无此数据。扩展数据源是最直接改进路径——这是数据获取问题，非建模问题。")
add_bullet("低级别联赛（英甲、英乙）预测精度较低，因比赛随机性更大。")
add_bullet("平局预测能力有限，模型仅 13.0%、市场 0.0%，说明平局本质上是最难分类的结果。")
add_bullet("投注 ROI 对阈值选择敏感（0%~100% 对应 +43%~+99%），分类准确率优势则稳定得多。")
add_bullet("未来方向：获取低级别联赛 PPG/xG 数据（最直接改进路径）、加入球员级特征（身价、伤病、阵容深度）。")

# ═════════════════════════════════════════════════════════
add_heading("六、参考文献", 1)
add_bullet("Buchdahl, J. (2003). Rating Systems for Fixed Odds Football Match Prediction.")
add_bullet("Reade, J., Singleton, C. & Vaughan Williams, L. (2020). Betting markets and statistical models. [模型 vs 市场比较方法论]")
add_bullet("Luiz et al. (2024). A deep learning approach for football match prediction.")
add_bullet("Atta Mills et al. (2024). [特征工程方法]")

doc.save(OUT)
print(f"Saved: {OUT}")

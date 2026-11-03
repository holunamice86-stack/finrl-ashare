# FinRL-DeepSeek · A股 AI 选股操作台

基于 [FinRL-DeepSeek](https://arxiv.org/abs/2502.07393)（CPPO + 大模型新闻信号）的 **A 股落地实现**：AI 选股 + 人择时的完整工具链。

> ⚠️ 本项目是**研究与教学演示**，不构成投资建议。模型输出的是「仓位偏好分布」，不是价格预测；A 股有 T+1、涨跌停、印花税等约束，实盘风险自负。

## 功能

| 模块 | 说明 |
|---|---|
| 🎯 AI 选股榜 | 一线/二线/三线股票池各自排行 + 市场环境仪表盘（择时由你决定） |
| 👁 AI 盯盘板 | 实时价格 + 模型实时偏好信号 + 动量外推（非预测） |
| ⑤ 买卖建议 | 期望收益排行、买入/卖出提醒、持仓监控、DeepSeek 人话解读 |
| 📰 新闻信号 | DeepSeek 对每日 A 股新闻打 1–5 情绪/风险分，喂给智能体 |
| 📈 虚拟模拟盘 | T+1 / 涨跌停 / 100股整手 / 印花税 0.05%，逐日执行 |
| 🗳 四模型投票集成 | 多个随机种子训练的模型共同打分取平均（降低单模型方差） |
| 🧪 训练/回测 | 网页一键训练 CVaR-PPO、验证集选模、留出段多路径评估 |

## 模型原理（一句话版）

智能体每天观察一个 **613 维状态**（现金 + 50 只股票的收盘价/持仓/8 个技术指标/新闻情绪/风险分 + 19 个市场指数特征），策略网络输出每只股票的**仓位偏好分布 μ/σ**，从分布采样得到实际买卖动作；目标是最大化长期资产收益，同时用 **CVaR** 约束惩罚极端亏损。DeepSeek 新闻信号以「轻量扰动」方式注入状态和动作（扰动幅度仅 5–10%，保证训练稳定）。

## 关键实验结果（留出段 2025-07 → 2026-10，20 条路径采样）

| 实验 | 收益 | 对照（等权持有） |
|---|---|---|
| 单种子（生产模型） | +45.86% | +16.56% |
| **多种子复现平均（3 个新种子）** | **+25.89%（±9.0p）** | +16.56% |
| 泛化：二线池（未见过的 50 只） | +107.59% | +69.68% |
| 泛化：创业板池 | +58.75% | +69.68% |

诚实结论：方向性（跑赢买入持有）经得起多种子复现；单次回测数字受种子随机性影响约 ±9 个百分点，引用时务必注明。

## 目录结构

```
finrl-ashare/
├── webapp/                  # Flask 网页操作台（waitress 生产服务器）
│   ├── server.py            # 全部 API（鉴权、秒回+轮询、NaN 消毒）
│   └── static/              # 前端（极简版 + 完整版）
├── train_web.py             # CVaR-PPO 训练入口
├── env_stocktrading_ashare.py  # A股环境: T+1/涨跌停/整手/印花税
├── make_ashare_data.py      # 新浪行情+资金流 → 训练CSV（含19个指数特征）
├── fetch_score_news.py      # 新闻抓取 + DeepSeek 打分
├── use_model.py             # 回测/评估（多路径采样、regime guard）
├── select_best.py           # 验证集选模
├── advice.py / live_panel.py / paper_trade.py / intraday_decision.py
└── nightly_seeds.sh         # 多种子复现一键流水线
```

## 快速开始

1. **安装依赖**
   ```bash
   python -m venv finrl_env && source finrl_env/bin/activate
   pip install -r requirements.txt
   # spinningup 需从源码安装（CVaR-PPO 训练核心）
   git clone https://github.com/openai/spinningup && pip install -e spinningup
   ```

2. **生成数据**（新浪财经国内接口，约 5 分钟）
   ```bash
   python make_ashare_data.py --idx2              # 一线池（沪深300前50 + 19指数特征）
   python make_ashare_data.py --idx2 --pool second50   # 二线池
   python make_ashare_data.py --idx2 --pool cyb50      # 三线池（创业板）
   python fetch_score_news.py --mode ashare_idx2  # 需 DEEPSEEK_API_KEY 环境变量
   ```

3. **训练**（约 2.5–3 小时 / 30 轮）
   ```bash
   export PYTHONPATH=$(pwd)/spinningup
   python train_web.py --epochs 30 --steps 20000 --data-csv ashare_idx2_2022_2026.csv \
       --data-start 2021-09-01 --data-end 2024-12-31 --model-name my_model --seed 0
   ```

4. **选模 + 留出评估**
   ```bash
   python select_best.py --name my_model --csv ashare_idx2_2022_2026.csv --paths 8
   python use_model.py --model trained_models/my_model_best.pth --data-csv ashare_idx2_2022_2026.csv \
       --data-start 2025-07-01 --eval-mode sample --eval-n 20
   ```

5. **启动网页操作台**
   ```bash
   cd webapp && python server.py     # 访问 http://127.0.0.1:8000
   ```

## 依赖与致谢

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)（MIT）—— 金融强化学习框架
- [FinRL-DeepSeek](https://arxiv.org/abs/2502.07393) —— CPPO + LLM 信号方法论
- [OpenAI SpinningUp](https://github.com/openai/spinningup)（MIT）—— PPO/CVaR-PPO 训练核心
- 数据：新浪财经公开接口；新闻打分：DeepSeek API

## 免责声明

本项目仅用于学习与研究。所有回测与模拟结果基于历史数据，不代表未来表现；模型输出是随机采样的仓位偏好，不是投资建议。

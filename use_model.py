# -*- coding: utf-8 -*-
"""
use_model.py —— 用训练好的 CPPO-DeepSeek 模型在 2019-2023 交易数据上做模拟交易
用法（在 WSL 里）:
    cd ~/FinRL_DeepSeek
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTHONPATH=~/spinningup:$PYTHONPATH
    ~/finrl_env/bin/python use_model.py
输出:
    results/回测_账户曲线.png   账户价值曲线（模型 vs 等权买入持有）
    results/回测_账户明细.csv   每日账户价值
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from datasets import load_dataset

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30",
              "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

# ---------- 网络结构（必须和训练时一致：512x512, Tanh） ----------
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="trained_models/agent_cppo_deepseek_100_epochs_20k_steps_99_101.pth")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--data-csv", type=str, default="", help="本地数据CSV(空=官方HF交易数据)")
    parser.add_argument("--data-start", type=str, default="", help="回测起始日期")
    parser.add_argument("--data-end", type=str, default="", help="回测结束日期")
    parser.add_argument("--eval-mode", type=str, default="mean", choices=["mean", "sample"],
                        help="评估方式: mean=确定性均值动作, sample=与训练一致的随机采样")
    parser.add_argument("--eval-n", type=int, default=5, help="采样评估时平均的路径数")
    parser.add_argument("--regime-guard", action="store_true",
                        help="择时罩子: 上证20日动量为负时削减买入")
    parser.add_argument("--regime-scale", type=float, default=0.3,
                        help="弱势市场买入缩放(0-1)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 1. 交易期数据（官方 HF 2019-2023 或本地近期 CSV） ----------
    if args.data_csv:
        print("加载本地数据 " + args.data_csv)
        trade = pd.read_csv(args.data_csv)
    else:
        print("下载交易数据 trade_data_deepseek_risk_2019_2023.csv ...")
        dataset = load_dataset("benstaf/nasdaq_2013_2023",
                               data_files="trade_data_deepseek_risk_2019_2023.csv")
        trade = pd.DataFrame(dataset["train"])
    trade = trade.drop([c for c in trade.columns if str(c).startswith("Unnamed")], axis=1)
    if args.data_start:
        trade = trade[trade["date"] >= args.data_start]
    if args.data_end:
        trade = trade[trade["date"] <= args.data_end]

    unique_dates = trade["date"].unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")

    trade["llm_sentiment"] = trade["llm_sentiment"].fillna(0)
    trade["llm_risk"] = trade["llm_risk"].fillna(3)

    stock_dim = len(trade.tic.unique())
    _idx_cols = [c for c in trade.columns if str(c).startswith('idx_')]
    state_space = (1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim
                   + len(_idx_cols))
    print(f"股票数 {stock_dim}, 状态维度 {state_space}, 交易日数 {len(unique_dates)}")

    # ---------- 2. 构建交易环境 ----------
    if args.data_csv and "ashare" in args.data_csv:
        from env_stocktrading_ashare import StockTradingEnv  # A股规则环境
        hmax = 10000
    else:
        from env_stocktrading_llm_risk import StockTradingEnv
        hmax = 100

    has_vix = "vix" in trade.columns
    env_kwargs = dict(
        hmax=hmax, initial_amount=1000000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        state_space=state_space, stock_dim=stock_dim,
        tech_indicator_list=INDICATORS, action_space=stock_dim,
        reward_scaling=1e-4,
    )
    if has_vix:
        env = StockTradingEnv(df=trade, turbulence_threshold=70,
                              risk_indicator_col="vix", **env_kwargs)
    else:
        env = StockTradingEnv(df=trade, **env_kwargs)

    # ---------- 3. 重建网络并加载训练好的权重 ----------
    ac = MLPActorCritic(obs_dim=state_space, act_dim=stock_dim,
                        hidden_sizes=list(args.hidden), activation=nn.Tanh)
    state_dict = torch.load(args.model, map_location="cpu")
    ac.load_state_dict(state_dict)
    ac.eval()
    print("模型加载成功:", args.model)

    _idx_len = len([c for c in trade.columns if str(c).startswith("idx_")])

    # ---------- 4. 逐日模拟交易 ----------
    def make_env():
        if has_vix:
            return StockTradingEnv(df=trade, turbulence_threshold=70,
                                   risk_indicator_col="vix", **env_kwargs)
        return StockTradingEnv(df=trade, **env_kwargs)

    def run_episode(mode):
        e = make_env()
        obs, _ = e.reset()
        dts, vals = [], []
        while True:
            obs_t = torch.as_tensor(np.array(obs, dtype=np.float32))
            with torch.no_grad():
                dist = ac.pi(obs_t)
                action = dist.mean if mode == "mean" else dist.sample()
            a_np = action.numpy()
            if args.regime_guard and _idx_len >= 3:
                mom20 = float(obs[-_idx_len + 2])  # idx_sh_mom20
                if mom20 <= -2.0:
                    buy_scale = 0.0   # 深度弱势: 禁止新买入
                elif mom20 < 0.0:
                    buy_scale = args.regime_scale  # 弱势: 买入打折
                else:
                    buy_scale = 1.0
                a_np = np.where(a_np > 0, a_np * buy_scale, a_np)  # 卖出不拦
            obs, reward, done, _, _ = e.step(a_np)
            if done:
                break
            # 当前总资产 = 现金 + 持仓市值
            prices = np.array(obs[1:1 + stock_dim])
            shares = np.array(obs[stock_dim + 1:stock_dim * 2 + 1])
            vals.append(obs[0] + float(np.sum(prices * shares)))
            dts.append(e.date_memory[-1])
        return e, dts, vals

    eval_n = max(1, args.eval_n)
    if args.eval_mode == "sample" and eval_n > 1:
        envs, all_vals = [], []
        for i in range(eval_n):
            e, d, v = run_episode("sample")
            envs.append(e)
            all_vals.append(v)
            if i == 0:
                dates = d
        account_values = np.array(all_vals).mean(axis=0).tolist()
        finals = [v[-1] for v in all_vals]
        total_cost = float(np.mean([e.cost for e in envs]))
        total_trades = int(np.mean([e.trades for e in envs]))
        env = envs[0]
        print("采样评估: %d 条路径, 期末范围 %s ~ %s" %
              (eval_n, f"{min(finals):,.0f}", f"{max(finals):,.0f}"))
    else:
        env, dates, account_values = run_episode("mean")
        total_cost, total_trades = env.cost, env.trades
        print("评估方式: 确定性均值动作")

    # ---------- 5. 等权买入持有基准（首日收盘价买入，无交易成本） ----------
    closes = np.array([env.state_memory[i][1:1 + stock_dim] for i in range(len(env.state_memory))])
    hold_values = np.sum((1_000_000 / stock_dim) * closes / closes[0], axis=1)

    # ---------- 6. 输出结果 ----------
    final = account_values[-1]
    summary = {
        "initial": 1_000_000,
        "final_value": round(float(final), 2),
        "hold_value": round(float(hold_values[-1]), 2),
        "return_pct": round((final / 1_000_000 - 1) * 100, 2),
        "hold_return_pct": round((hold_values[-1] / 1_000_000 - 1) * 100, 2),
        "total_cost": round(float(total_cost), 2),
        "total_trades": int(total_trades),
        "days": int(len(dates)),
        "eval_mode": args.eval_mode,
        "eval_n": eval_n,
    }
    with open(os.path.join(args.outdir, "回测_结果.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n========== 模拟交易结果 ==========")
    print(f"初始资金      : 1,000,000")
    print(f"最终总资产    : {final:,.0f}")
    print(f"总收益率      : {(final / 1_000_000 - 1) * 100:.2f}%")
    print(f"等权买入持有  : {hold_values[-1]:,.0f} ({(hold_values[-1] / 1_000_000 - 1) * 100:.2f}%)")
    print(f"累计交易成本  : {total_cost:,.0f}")
    print(f"总交易次数    : {total_trades}")
    print("==================================\n")

    df_out = pd.DataFrame({"date": dates, "account_value": account_values,
                           "hold_value": hold_values})
    df_out.to_csv(os.path.join(args.outdir, "回测_账户明细.csv"), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df_out["date"], df_out["account_value"], label="CPPO-DeepSeek agent")
    plt.plot(df_out["date"], df_out["hold_value"], label="Equal-weight buy & hold")
    plt.title("FinRL-DeepSeek backtest 2019-2023")
    plt.xlabel("Date")
    plt.ylabel("Account value (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "回测_账户曲线.png"), dpi=120)
    print("结果已保存到:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()

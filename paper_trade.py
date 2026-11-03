# -*- coding: utf-8 -*-
"""
paper_trade.py —— A股虚拟模拟盘引擎（无需券商账户）

每个交易日运行一次：
1. 读最新行情（ashare CSV 最后一天）
2. 用训练好的模型生成动作（随机采样，与训练分布一致）
3. 按 A 股规则虚拟成交：T+1 / 涨跌停 / 100股整手 / 卖出印花税0.05%
4. 更新虚拟账户（现金/持仓/可卖份额），记录每日净值

账户文件: ~/FinRL_DeepSeek/paper_account_ashare.json
用法: ~/finrl_env/bin/python paper_trade.py --json   （--reset 重置账户）
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

BASE = os.path.expanduser("~/FinRL_DeepSeek")
CSV = os.path.join(BASE, "ashare_2022_2026.csv")
ACCOUNT = os.path.join(BASE, "paper_account_ashare.json")
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
              "close_30_sma", "close_60_sma"]
HMAX = 10000
INIT_CASH = 1_000_000


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32)))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        return Normal(self.mu_net(obs), torch.exp(self.log_std))


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)


def limit_pct(code):
    code = str(code).zfill(6)
    return 0.20 if code.startswith(("688", "300", "301")) else 0.10


def build_state(day, tics, cash, shares, sents, risks):
    closes = day["close"].astype(float).tolist()
    inds = []
    for ind in INDICATORS:
        inds += day[ind].astype(float).fillna(0).tolist()
    idx_cols = [c for c in day.columns if str(c).startswith("idx_")]
    idx_vals = day[idx_cols].iloc[0].fillna(0).tolist() if idx_cols else []
    return [cash] + closes + shares + inds + sents + risks + idx_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="只输出 JSON（供网页调用）")
    parser.add_argument("--reset", action="store_true", help="重置虚拟账户")
    parser.add_argument("--model", type=str,
                        default=os.path.join(BASE, "trained_models", "ashare_30ep_v2_best.pth"))
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--csv", type=str, default=None, help="数据 CSV（默认主池）")
    args = parser.parse_args()

    if args.reset and os.path.exists(ACCOUNT):
        os.remove(ACCOUNT)
        print(json.dumps({"ok": True, "msg": "模拟盘已重置"}) if args.json else "模拟盘已重置")
        return

    df = pd.read_csv(args.csv or CSV)
    last_date = df["date"].max()
    day = df[df["date"] == last_date].sort_values("tic")
    tics = [str(t).zfill(6) for t in day["tic"].tolist()]
    D = len(tics)
    prev_date = df[df["date"] < last_date]["date"].max()
    prev = df[df["date"] == prev_date].sort_values("tic")
    prev_close = dict(zip([str(t).zfill(6) for t in prev["tic"].tolist()],
                          prev["close"].astype(float).tolist()))

    # ---- 账户加载或初始化 ----
    if os.path.exists(ACCOUNT):
        with open(ACCOUNT, encoding="utf-8") as f:
            acc = json.load(f)
    else:
        acc = {"cash": INIT_CASH, "last_date": None, "model": os.path.basename(args.model),
               "positions": {t: {"shares": 0, "sellable": 0, "today_bought": 0, "cost": 0} for t in tics},
               "history": []}
    if acc.get("last_date") == str(last_date):
        msg = "今日决策已执行过（%s），明天再来 | 当前模型 %s" % (last_date, os.path.basename(args.model))
        print(json.dumps({"ok": False, "msg": msg}) if args.json else msg)
        return

    # 昨日买入 → 今日可卖（T+1 滚动）
    for t in tics:
        p = acc["positions"].setdefault(t, {"shares": 0, "sellable": 0, "today_bought": 0, "cost": 0})
        p["sellable"] += p.get("today_bought", 0)
        p["today_bought"] = 0

    # ---- 模型动作（采样，与训练分布一致） ----
    ac = MLPActorCritic(obs_dim=1 + 2 * D + (2 + len(INDICATORS)) * D +
                        len([c for c in day.columns if str(c).startswith("idx_")]),
                        act_dim=D, hidden_sizes=list(args.hidden))
    ac.load_state_dict(torch.load(args.model, map_location="cpu"))
    ac.eval()
    closes = day["close"].astype(float).tolist()
    shares = [acc["positions"][t]["shares"] for t in tics]
    sents = day["llm_sentiment"].astype(float).fillna(3).tolist()
    risks = day["llm_risk"].astype(float).fillna(3).tolist()
    state = build_state(day, tics, acc["cash"], shares, sents, risks)
    with torch.no_grad():
        a = ac.pi(torch.as_tensor(np.array(state, dtype=np.float32))).sample().numpy()

    # ---- 虚拟成交（A股规则 + 单票仓位约束） ----
    POS_CAP = 0.20  # 单票市值不超过总资产的 20%（风控：防集中持仓）
    value0 = acc["cash"] + sum(closes[i] * acc["positions"][t]["shares"]
                               for i, t in enumerate(tics))
    orders, cost = [], 0.0
    for i, tic in enumerate(tics):
        price = closes[i]
        raw = int(round(a[i] * HMAX))
        pct = limit_pct(tic)
        up = tic in prev_close and price >= round(prev_close[tic] * (1 + pct), 2) - 1e-6
        down = tic in prev_close and price <= round(prev_close[tic] * (1 - pct), 2) + 1e-6
        p = acc["positions"][tic]
        if raw > 0 and not up:
            lots = (raw // 100) * 100
            afford = int((acc["cash"] // (price * 1.001)) // 100) * 100
            lots = min(lots, afford)
            # 仓位约束: 买入后该股市值 <= 总资产 * 20%
            room = max(0.0, value0 * POS_CAP - price * p["shares"])
            room_lots = int((room / (price * 1.001)) // 100) * 100
            lots = min(lots, room_lots)
            if lots > 0:
                amount = price * lots * 1.001
                acc["cash"] -= amount
                p["shares"] += lots
                p["today_bought"] += lots
                # 摊薄成本价（含佣金）
                p["cost"] = (p.get("cost", 0) * (p["shares"] - lots) + price * lots * 1.001) / p["shares"]
                cost += price * lots * 0.001
                orders.append({"tic": tic, "side": "买", "shares": lots, "price": round(price, 2)})
        elif raw < 0 and not down and p["sellable"] > 0:
            n = min(abs(raw), p["sellable"])
            if n > 0:
                amount = price * n * (1 - 0.001 - 0.0005)  # 佣金+印花税
                acc["cash"] += amount
                p["shares"] -= n
                p["sellable"] -= n
                cost += price * n * (0.001 + 0.0005)
                orders.append({"tic": tic, "side": "卖", "shares": n, "price": round(price, 2)})

    value = acc["cash"] + sum(closes[i] * acc["positions"][t]["shares"] for i, t in enumerate(tics))
    acc["last_date"] = str(last_date)
    acc["model"] = os.path.basename(args.model)  # 每次执行都记录实际使用的模型
    acc["history"].append({
        "date": str(last_date), "value": round(value, 2), "cash": round(acc["cash"], 2),
        "orders": len(orders), "cost": round(cost, 2),
    })
    with open(ACCOUNT, "w", encoding="utf-8") as f:
        json.dump(acc, f, ensure_ascii=False, indent=2)

    # 净值曲线（有历史就出图，单日也能画）
    if len(acc["history"]) >= 1:
        h = acc["history"]
        plt.figure(figsize=(10, 5))
        plt.plot([x["date"] for x in h], [x["value"] for x in h], marker="o")
        plt.title("Paper Account Value (A-share)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        img = os.path.join(BASE, "webapp", "static", "results", "paper", "paper_curve.png")
        os.makedirs(os.path.dirname(img), exist_ok=True)
        plt.savefig(img, dpi=100)
        plt.close()

    ret = (value / INIT_CASH - 1) * 100
    top = sorted(((t, acc["positions"][t]["shares"]) for t in tics
                  if acc["positions"][t]["shares"] > 0), key=lambda x: -x[1])[:10]
    out = {"ok": True, "date": str(last_date), "cash": round(acc["cash"], 2),
           "value": round(value, 2), "return_pct": round(ret, 2),
           "orders": orders, "top_positions": [{"tic": t, "shares": s} for t, s in top],
           "days": len(acc["history"]), "pos_cap": POS_CAP,
           "img": "/static/results/paper/paper_curve.png"}
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print("=== 模拟盘 %s ===" % last_date)
        print("现金 %.0f | 总资产 %.0f (%.2f%%) | 今日订单 %d 笔" %
              (acc["cash"], value, ret, len(orders)))
        for o in orders[:10]:
            print("  %s %s %d股 @%.2f" % (o["tic"], o["side"], o["shares"], o["price"]))


if __name__ == "__main__":
    main()

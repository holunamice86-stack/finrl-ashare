# -*- coding: utf-8 -*-
"""
intraday_decision.py —— 盘中/尾盘实时决策（方案A）

用途: 交易日 14:30-15:00 之间运行，用新浪实时行情构造「准今日K线」，
让模型基于实时价 + 模拟盘实际持仓(T+1可卖份额)给出可执行的买卖建议，
供你在收盘前手动下单。收盘后再运行则退化为标准的收盘决策。

用法: ~/finrl_env/bin/python intraday_decision.py --json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from stockstats import StockDataFrame as Sdf

BASE = os.path.expanduser("~/FinRL_DeepSeek")
CSV = os.path.join(BASE, "ashare_2022_2026.csv")  # 生产主池(可被 --csv 覆盖)
ACCOUNT = os.path.join(BASE, "paper_account_ashare.json")
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
              "close_30_sma", "close_60_sma"]
HMAX = 10000
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}


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


def sina_symbol(code):
    code = str(code).zfill(6)
    return ("sh" if code[0] in "69" else "sz") + code


def limit_pct(code):
    code = str(code).zfill(6)
    return 0.20 if code.startswith(("688", "300", "301")) else 0.10


def fetch_realtime(tics):
    """返回 {tic: {open, high, low, close, volume, date}}，停牌股缺席"""
    url = "https://hq.sinajs.cn/list=" + ",".join(sina_symbol(t) for t in tics)
    r = requests.get(url, timeout=15, headers=HEADERS)
    r.encoding = "gbk"
    out = {}
    for line in r.text.strip().splitlines():
        if '="' not in line:
            continue
        head, body = line.split('="', 1)
        sym = head.split("_")[-1].strip()
        code = sym[2:]
        f = body.rstrip('";').split(",")
        if len(f) < 32 or not f[3]:
            continue
        out[code] = {"open": float(f[1]), "prev_close": float(f[2]),
                     "close": float(f[3]), "high": float(f[4]), "low": float(f[5]),
                     "volume": float(f[8]), "date": f[30]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", type=str, default=None, help="数据 CSV")
    parser.add_argument("--model", type=str,
                        default=os.path.join(BASE, "trained_models", "ashare_30ep_v2_best.pth"))
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    args = parser.parse_args()

    df = pd.read_csv(args.csv or CSV)
    tics = [str(t).zfill(6) for t in sorted(df["tic"].astype(str).str.zfill(6).unique())]
    D = len(tics)
    last_date = df["date"].max()
    rt = fetch_realtime(tics)
    rt_date = max((v["date"] for v in rt.values()), default="")

    if str(rt_date) == str(last_date):
        mode = "收盘后（用当日收盘数据）"
        day = df[df["date"] == last_date].sort_values("tic")
    else:
        mode = "盘中实时价（准今日K线，越接近收盘越准）"
        # 构造今日代理行: 实时 open/high/low/close(现价)/volume
        tail = df[df["date"] >= df["date"].max() - pd.Timedelta(days=0)]
        # 取每只股票最近 400 行 + 追加今日代理行，重算指标
        parts = []
        for t in tics:
            g = df[df["tic"].astype(str).str.zfill(6) == t].tail(400).copy()
            if t in rt:
                row = {"date": rt[t]["date"], "tic": t, "open": rt[t]["open"],
                       "high": rt[t]["high"], "low": rt[t]["low"],
                       "close": rt[t]["close"], "volume": rt[t]["volume"]}
            else:  # 停牌: 用昨日收盘价冻结
                lastrow = g.iloc[-1]
                row = {"date": rt_date or str(last_date), "tic": t,
                       "open": lastrow["close"], "high": lastrow["close"],
                       "low": lastrow["close"], "close": lastrow["close"],
                       "volume": 0}
            parts.append(pd.concat([g, pd.DataFrame([row])], ignore_index=True))
        ext = pd.concat(parts, ignore_index=True)
        ext = ext.sort_values(["tic", "date"]).reset_index(drop=True)
        stock = Sdf.retype(ext.copy())
        rows = []
        for t in tics:
            s = stock[stock["tic"].astype(str).str.zfill(6) == t]
            lastrow = ext[(ext["tic"].astype(str).str.zfill(6) == t)].iloc[-1].copy()
            for col in INDICATORS:
                lastrow[col] = s[col].values[-1]
            rows.append(lastrow)
        day = pd.DataFrame(rows).sort_values("tic")
        # 今日新闻打分（有就用）
        news_today = df[df["date"].astype(str) == str(rt_date)]
        if len(news_today):
            m = dict(zip(news_today["tic"].astype(str).str.zfill(6),
                         zip(news_today["llm_sentiment"], news_today["llm_risk"])))
            day["llm_sentiment"] = [m.get(t, (0.0, 3.0))[0] for t in tics]
            day["llm_risk"] = [m.get(t, (0.0, 3.0))[1] for t in tics]

    # 模拟盘账户
    with open(ACCOUNT, encoding="utf-8") as f:
        acc = json.load(f)
    closes = day["close"].astype(float).tolist()
    shares = [acc["positions"].get(t, {}).get("shares", 0) for t in tics]
    sellable = [acc["positions"].get(t, {}).get("sellable", 0) for t in tics]
    sents = day["llm_sentiment"].astype(float).fillna(3).tolist()
    risks = day["llm_risk"].astype(float).fillna(3).tolist()
    inds = []
    for ind in INDICATORS:
        inds += day[ind].astype(float).fillna(0).tolist()
    idx_cols = [c for c in day.columns if str(c).startswith("idx_")]
    state = ([acc["cash"]] + closes + shares + inds + sents + risks +
             (day[idx_cols].iloc[0].fillna(0).tolist() if idx_cols else []))

    ac = MLPActorCritic(obs_dim=1 + 2 * D + (2 + len(INDICATORS)) * D + len(idx_cols),
                        act_dim=D, hidden_sizes=list(args.hidden))
    ac.load_state_dict(torch.load(args.model, map_location="cpu"))
    ac.eval()
    with torch.no_grad():
        a = ac.pi(torch.as_tensor(np.array(state, dtype=np.float32))).sample().numpy()

    # 建议订单（尊重 T+1 可卖份额与现金约束）
    prev_close = dict(zip(tics, (rt.get(t, {}).get("prev_close") for t in tics)))
    buys, sells = [], []
    for i, t in enumerate(tics):
        price = closes[i]
        pct = limit_pct(t)
        pc = prev_close.get(t)
        up = pc and price >= round(pc * (1 + pct), 2) - 1e-6
        down = pc and price <= round(pc * (1 - pct), 2) + 1e-6
        raw = int(round(a[i] * HMAX))
        if raw > 0 and not up:
            lots = min((raw // 100) * 100, int(acc["cash"] // (price * 1.001) // 100) * 100)
            if lots > 0:
                buys.append({"tic": t, "shares": lots, "price": round(price, 2)})
        elif raw < 0 and not down and sellable[i] > 0:
            n = min(abs(raw), sellable[i])
            if n > 0:
                sells.append({"tic": t, "shares": n, "price": round(price, 2)})

    out = {"date": str(rt_date or last_date), "mode": mode, "model": os.path.basename(args.model),
           "buys": sorted(buys, key=lambda x: -x["shares"])[:8],
           "sells": sorted(sells, key=lambda x: -x["shares"])[:8],
           "buy_count": len(buys), "sell_count": len(sells)}
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print("=== %s (%s) ===" % (out["date"], mode))
        print("建议买入:", ", ".join("%s %d股@%.2f" % (b["tic"], b["shares"], b["price"]) for b in out["buys"]) or "无")
        print("建议卖出:", ", ".join("%s %d股@%.2f" % (s["tic"], s["shares"], s["price"]) for s in out["sells"]) or "无")


if __name__ == "__main__":
    main()

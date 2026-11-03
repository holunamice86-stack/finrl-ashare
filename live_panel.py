# -*- coding: utf-8 -*-
"""live_panel.py —— AI 盯盘板后端
输入: --tics 600519,300750 (2-3只)
输出: 每只股票的 5分钟K线 + 实时价 + 模型实时偏好分(μ) + 动量外推虚线(诚实标注)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import requests
import torch

from intraday_decision import (mlp, MLPActorCritic, fetch_realtime, INDICATORS)

BASE = os.path.expanduser("~/FinRL_DeepSeek")
CSV = os.path.join(BASE, "ashare_idx2_2022_2026.csv")
ACCOUNT = os.path.join(BASE, "paper_account_ashare.json")
HMAX = 10000
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}


def sina_symbol(code):
    code = str(code).zfill(6)
    return ("sh" if code[0] in "69" else "sz") + code


def get_minute_bars(symbol, scale=5, datalen=60):
    """5分钟K线（新浪），返回 [(时间, 收盘)] 列表"""
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%%20_=/"
           "CN_MarketDataService.getKLineData?symbol=%s&scale=%d&ma=no&datalen=%d"
           % (symbol, scale, datalen))
    r = requests.get(url, timeout=12, headers=HEADERS)
    t = r.text
    i, j = t.find("(["), t.rfind("])")
    if i == -1 or j == -1:
        return []
    arr = json.loads(t[i + 1:j + 1])
    return [[x["day"], float(x["close"])] for x in arr]


def signal_of(mu):
    if mu >= 0.3:
        return "强烈买入"
    if mu >= 0.05:
        return "买入"
    if mu <= -0.3:
        return "强烈卖出"
    if mu <= -0.05:
        return "卖出"
    return "持有"


def extrapolate(bars, horizon=6):
    """最近6根5分钟K线线性外推未来 3/6 根（15/30分钟），非预测"""
    if len(bars) < 6:
        return []
    closes = [b[1] for b in bars[-6:]]
    xs = np.arange(6, dtype=float)
    k = np.polyfit(xs, closes, 1)[0]
    last_t = bars[-1][0]
    out = []
    try:
        base = pd.Timestamp(last_t)
        for step in (3, 6):
            t = base + pd.Timedelta(minutes=5 * step)
            out.append([t.strftime("%H:%M"), round(closes[-1] + k * step, 2)])
    except Exception:
        pass
    return out


def build_day(df, tics, rt, rt_date, last_date):
    """盘中: 实时价构造准今日K线并重算指标; 收盘后: 直接用当日行"""
    from stockstats import StockDataFrame as Sdf
    if str(rt_date) > str(last_date):
        rows = []
        for t in tics:
            r = rt.get(t, {})
            ext = df[df["tic"].astype(str).str.zfill(6) == t].tail(400)
            if ext.empty:
                rows.append({"tic": t})
                continue
            lastrow = ext.iloc[-1].copy()
            if r.get("price"):
                lastrow["date"] = rt_date
                lastrow["open"] = r.get("open") or r["price"]
                lastrow["high"] = r.get("high") or r["price"]
                lastrow["low"] = r.get("low") or r["price"]
                lastrow["close"] = r["price"]
                lastrow["volume"] = r.get("volume", 0) or 0
            rows.append(lastrow)  # 无实时价时回退到最后一根已知K线
        day = pd.DataFrame(rows)
        if day.empty or "close" not in day.columns:
            day = df[df["date"].astype(str) == str(last_date)].copy()
            day["tic"] = day["tic"].astype(str).str.zfill(6)
            return day.sort_values("tic"), "收盘数据"
        if "close" in day.columns:
            stock = Sdf.retype(day.copy())
            for col in INDICATORS:
                vals = []
                for t in tics:
                    s = stock[stock["tic"].astype(str).str.zfill(6) == t]
                    vals.append(s[col].values[-1] if len(s) else 0)
                day[col] = vals
        news_today = df[df["date"].astype(str) == str(rt_date)]
        if len(news_today):
            m = dict(zip(news_today["tic"].astype(str).str.zfill(6),
                         zip(news_today["llm_sentiment"], news_today["llm_risk"])))
            day["llm_sentiment"] = [m.get(t, (0.0, 3.0))[0] for t in tics]
            day["llm_risk"] = [m.get(t, (0.0, 3.0))[1] for t in tics]
        return day, "盘中实时价"
    day = df[df["date"].astype(str) == str(last_date)].copy()
    day["tic"] = day["tic"].astype(str).str.zfill(6)
    day = day.sort_values("tic")
    return day, "收盘数据"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tics", type=str, default="")
    parser.add_argument("--model", type=str,
                        default=os.path.join(BASE, "trained_models",
                                             "ashare_idx2_30ep_best.pth"))
    parser.add_argument("--models", type=str, default=None,
                        help="多模型投票集成: 逗号分隔文件名, μ取平均")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    want = [str(t).strip().zfill(6) for t in args.tics.split(",") if t.strip()][:3]
    if not want:
        print(json.dumps({"ok": False, "msg": "未指定股票"}))
        return

    df = pd.read_csv(CSV)
    tics = [str(t).zfill(6) for t in sorted(df["tic"].astype(str).str.zfill(6).unique())]
    D = len(tics)
    last_date = df["date"].max()
    rt = fetch_realtime(tics)
    rt_date = max((v["date"] for v in rt.values()), default="")
    day, mode = build_day(df, tics, rt, rt_date, last_date)

    with open(ACCOUNT, encoding="utf-8") as f:
        acc = json.load(f)
    closes = day["close"].astype(float).tolist()
    shares = [acc["positions"].get(t, {}).get("shares", 0) for t in tics]
    sents = day["llm_sentiment"].astype(float).fillna(3).tolist()
    risks = day["llm_risk"].astype(float).fillna(3).tolist()
    inds = []
    for ind in INDICATORS:
        inds += day[ind].astype(float).fillna(0).tolist()
    idx_cols = [c for c in day.columns if str(c).startswith("idx_")]
    idx_vals = day[idx_cols].iloc[0].fillna(0).tolist() if idx_cols else []
    state = [acc["cash"]] + closes + shares + inds + sents + risks + idx_vals

    ac = MLPActorCritic(obs_dim=1 + 2 * D + (2 + len(INDICATORS)) * D + len(idx_cols),
                        act_dim=D, hidden_sizes=list(args.hidden))
    model_paths = [args.model]
    if args.models:
        model_paths = [os.path.join(BASE, "trained_models", m.strip())
                       for m in args.models.split(",") if m.strip()]
    mus = []
    for mp in model_paths:
        ac.load_state_dict(torch.load(mp, map_location="cpu"))
        ac.eval()
        with torch.no_grad():
            mus.append(ac.pi(torch.as_tensor(
                np.array(state, dtype=np.float32))).mean.numpy())
    mu = np.mean(mus, axis=0)  # 投票集成

    out_tics = []
    for t in want:
        if t not in tics:
            out_tics.append({"tic": t, "in_pool": False})
            continue
        i = tics.index(t)
        bars = get_minute_bars(sina_symbol(t))
        r = rt.get(t, {})
        price = r.get("price") or (closes[i] if i < len(closes) else None)
        chg = round((price / r["prev_close"] - 1) * 100, 2) if price and r.get("prev_close") else None
        out_tics.append({
            "tic": t, "in_pool": True,
            "price": price, "chg": chg,
            "mu": round(float(mu[i]), 3),
            "signal": signal_of(float(mu[i])),
            "bars": bars[-48:],
            "extrap": extrapolate(bars),
            "session": str(rt_date), "mode": mode,
        })
    print(json.dumps({"ok": True, "tics": out_tics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

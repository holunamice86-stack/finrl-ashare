# -*- coding: utf-8 -*-
"""
advice.py —— 统一建议生成器

输出三块内容（--json 供网页调用）:
1. ranking: 期望收益排行 —— 按模型期望动作 μ 排序（策略置信度近似），附 20 日动量参考
2. buy_orders / sell_orders: 可执行订单（采样动作 + A股规则 + 模拟盘 T+1 可卖份额/现金约束）
3. positions: 持仓监控 —— 每只持仓股: 成本/现价/盈亏% / 模型信号(μ方向) / 建议卖出股数

用法: ~/finrl_env/bin/python advice.py --json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import requests
import torch

from intraday_decision import (mlp, MLPActorCritic, sina_symbol, limit_pct,
                               fetch_realtime, INDICATORS)

BASE = os.path.expanduser("~/FinRL_DeepSeek")
CSV = os.path.join(BASE, "ashare_2022_2026.csv")
ACCOUNT = os.path.join(BASE, "paper_account_ashare.json")
KEY_PATH = os.path.join(BASE, ".deepseek_key")
HMAX = 10000


def load_api_key():
    try:
        with open(KEY_PATH) as f:
            return f.read().strip()
    except Exception:
        return ""


def rule_interpret(ranking, buys, sells, positions, date, capital):
    """内置规则解读（LLM 不可用时兜底）"""
    lines = []
    if ranking:
        top = ranking[0]
        lines.append("模型当前最看好 %s（偏好分 %.2f，全场第一）" % (top.get("name") or top["tic"], top["score"]))
        if top.get("chg") is not None:
            lines.append("它今日%s%.2f%%" % ("上涨" if top["chg"] >= 0 else "下跌", abs(top["chg"])))
        if top.get("sent") and top["sent"] >= 4:
            lines.append("且新闻面偏正面（%s分），属于信号共振" % top["sent"])
        elif top.get("sent") and top["sent"] <= 2:
            lines.append("但新闻面偏负面（%s分），模型疑似在逆势布局，注意风险" % top["sent"])
    if buys:
        lines.append("可执行买入建议：" + "、".join("%s %d股" % (b.get("name") or b["tic"], b["shares"]) for b in buys[:3]))
    if sells:
        lines.append("建议卖出：" + "、".join("%s %d股" % (s.get("name") or s["tic"], s["shares"]) for s in sells[:3]))
    else:
        lines.append("暂无卖出建议")
    if positions:
        neg = [p for p in positions if "卖" in p.get("signal", "")]
        pos = [p for p in positions if "买" in p.get("signal", "")]
        if neg:
            lines.append("持仓里 %d 只出现卖出信号（%s），重点盯" % (len(neg), "、".join((p.get("name") or p["tic"]) for p in neg[:3])))
        if pos:
            lines.append("%d 只出现买入信号（%s），可继续持有" % (len(pos), "、".join((p.get("name") or p["tic"]) for p in pos[:3])))
    lines.append("（以上为模型偏好解读，非投资建议；建议股数来自随机采样，会随刷新略有变化）")
    return "。".join(lines) + "。"


def llm_interpret(summary):
    """用 DeepSeek 把模型输出翻译成人话；失败返回 None"""
    key = load_api_key()
    if not key:
        return None
    prompt = ("你是资深的中文A股投资顾问。下面是一个强化学习模型对50只A股的最新输出"
              "（评分=模型想买某只股票的强度，正=看涨偏好；新闻分1-5，1负面5正面；"
              "风险分1-5，越高越危险）。请用 4-6 句通俗大白话解读："
              "1) 模型现在整体最看好谁、为什么（结合今日涨跌/20日动量/新闻面）；"
              "2) 该买什么、该卖什么（股数来自随机采样，提醒一句股数会抖）；"
              "3) 持仓里最该注意的风险。不要列表、不要术语、不要提『模型输出』四个字，"
              "像和客户聊天一样说。数据：\n" + json.dumps(summary, ensure_ascii=False))
    try:
        r = requests.post("https://api.deepseek.com/chat/completions",
                          headers={"Authorization": "Bearer " + key,
                                   "Content-Type": "application/json"},
                          json={"model": "deepseek-chat",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 500, "temperature": 0.6},
                          timeout=40)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


def build_today_df(df, tics, rt, rt_date, last_date):
    """盘中: 实时价构造准今日K线并重算指标; 收盘后: 直接用当日行"""
    if str(rt_date) == str(last_date):
        return df[df["date"] == last_date].sort_values("tic"), "收盘后（当日收盘数据）"
    from stockstats import StockDataFrame as Sdf
    parts = []
    for t in tics:
        g = df[df["tic"].astype(str).str.zfill(6) == t].tail(400).copy()
        if t in rt:
            row = {"date": rt[t]["date"], "tic": t, "open": rt[t]["open"],
                   "high": rt[t]["high"], "low": rt[t]["low"],
                   "close": rt[t]["close"], "volume": rt[t]["volume"]}
        else:
            lastrow = g.iloc[-1]
            row = {"date": rt_date or str(last_date), "tic": t,
                   "open": lastrow["close"], "high": lastrow["close"],
                   "low": lastrow["close"], "close": lastrow["close"], "volume": 0}
        parts.append(pd.concat([g, pd.DataFrame([row])], ignore_index=True))
    ext = pd.concat(parts, ignore_index=True).sort_values(["tic", "date"]).reset_index(drop=True)
    stock = Sdf.retype(ext.copy())
    rows = []
    for t in tics:
        s = stock[stock["tic"].astype(str).str.zfill(6) == t]
        lastrow = ext[ext["tic"].astype(str).str.zfill(6) == t].iloc[-1].copy()
        for col in INDICATORS:
            lastrow[col] = s[col].values[-1]
        rows.append(lastrow)
    day = pd.DataFrame(rows).sort_values("tic")
    news_today = df[df["date"].astype(str) == str(rt_date)]
    if len(news_today):
        m = dict(zip(news_today["tic"].astype(str).str.zfill(6),
                     zip(news_today["llm_sentiment"], news_today["llm_risk"])))
        day["llm_sentiment"] = [m.get(t, (0.0, 3.0))[0] for t in tics]
        day["llm_risk"] = [m.get(t, (0.0, 3.0))[1] for t in tics]
    return day, "盘中实时价（准今日K线，越接近收盘越准）"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model", type=str,
                        default=os.path.join(BASE, "trained_models", "ashare_30ep_v2_best.pth"))
    parser.add_argument("--models", type=str, default=None,
                        help="多模型投票集成: 逗号分隔的模型文件名(trained_models下), μ与动作取平均")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--capital", type=float, default=0.0,
                        help="实盘可投入资金(元), 0=跟随模拟盘现金")
    parser.add_argument("--csv", type=str, default=None, help="数据 CSV（默认主池）")
    parser.add_argument("--no-llm", action="store_true", help="跳过自然语言解读（分层推荐用）")
    args = parser.parse_args()

    df = pd.read_csv(args.csv or CSV)
    tics = [str(t).zfill(6) for t in sorted(df["tic"].astype(str).str.zfill(6).unique())]
    D = len(tics)
    last_date = df["date"].max()
    rt = fetch_realtime(tics)
    rt_date = max((v["date"] for v in rt.values()), default="")
    day, mode = build_today_df(df, tics, rt, rt_date, last_date)

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
    state = ([acc["cash"]] + closes + shares + inds + sents + risks +
             (day[[c for c in day.columns if str(c).startswith("idx_")]].iloc[0].fillna(0).tolist()
              if any(str(c).startswith("idx_") for c in day.columns) else []))

    ac = MLPActorCritic(obs_dim=1 + 2 * D + (2 + len(INDICATORS)) * D +
                        len([c for c in day.columns if str(c).startswith("idx_")]),
                        act_dim=D, hidden_sizes=list(args.hidden))
    model_paths = [args.model]
    if args.models:
        model_paths = [os.path.join(BASE, "trained_models", m.strip())
                       for m in args.models.split(",") if m.strip()]
    mus, acts = [], []
    for mp in model_paths:
        ac.load_state_dict(torch.load(mp, map_location="cpu"))
        ac.eval()
        with torch.no_grad():
            dist = ac.pi(torch.as_tensor(np.array(state, dtype=np.float32)))
            mus.append(dist.mean.numpy())
            acts.append(dist.sample().numpy())
    mu = np.mean(mus, axis=0)      # 投票集成: 各模型偏好取平均
    a = np.mean(acts, axis=0)      # 动作也取平均(降低单模型采样噪声)
    n_models = len(model_paths)

    # 20 日动量（每只股票 20 个交易日前收盘价）
    mom = {}
    for t in tics:
        g = df[df["tic"].astype(str).str.zfill(6) == t]["close"].astype(float)
        if len(g) > 20:
            mom[t] = round((closes[tics.index(t)] / g.iloc[-21] - 1) * 100, 2)
        else:
            mom[t] = None

    # 1) 期望收益排行（μ 降序）
    order = np.argsort(-mu)
    ranking = []
    for i in order[:10]:
        t = tics[i]
        pc = rt.get(t, {}).get("prev_close")
        chg = round((closes[i] / pc - 1) * 100, 2) if pc else None
        ranking.append({"tic": t, "score": round(float(mu[i]), 3),
                        "mom20": mom.get(t), "close": round(closes[i], 2),
                        "chg": chg,
                        "sent": float(sents[i]), "risk": float(risks[i])})

    # 2) 可执行订单（采样动作 + 规则 + 单票仓位约束 20%）
    POS_CAP = 0.20
    use_budget = args.capital > 0
    budget_left = args.capital if use_budget else acc["cash"]
    value0 = acc["cash"] + sum(closes[i] * acc["positions"].get(t, {}).get("shares", 0)
                               for i, t in enumerate(tics))
    prev_close = dict(zip(tics, (rt.get(t, {}).get("prev_close") for t in tics)))
    buys, sells = [], []
    # 买入：按动作强度从大到小依次分配资金
    for i in sorted([i for i in range(len(a)) if a[i] > 0], key=lambda i: -a[i]):
        t = tics[i]
        price = closes[i]
        pc = prev_close.get(t)
        up = pc and price >= round(pc * (1 + limit_pct(t)), 2) - 1e-6
        if up:
            continue
        if price * 100 * 1.001 > budget_left:
            continue  # 一手都买不起
        desired = int(a[i] * HMAX * (args.capital / 1e6)) if use_budget else int(a[i] * HMAX)
        lots = min((desired // 100) * 100,
                   int(budget_left // (price * 1.001) // 100) * 100)
        base = args.capital if use_budget else value0
        room = max(0.0, base * POS_CAP -
                   price * acc["positions"].get(t, {}).get("shares", 0))
        lots = min(lots, int((room / (price * 1.001)) // 100) * 100)
        if lots > 0:
            buys.append({"tic": t, "shares": lots, "price": round(price, 2)})
            budget_left -= price * lots * 1.001
    # 卖出：跟随模拟盘可卖份额
    for i, t in enumerate(tics):
        price = closes[i]
        pc = prev_close.get(t)
        down = pc and price <= round(pc * (1 + limit_pct(t)), 2) + 1e-6
        raw = int(round(a[i] * HMAX))
        if raw < 0 and not down and sellable[i] > 0:
            n = min(abs(raw), sellable[i])
            if n > 0:
                sells.append({"tic": t, "shares": n, "price": round(price, 2)})
    buys.sort(key=lambda x: -x["shares"])
    sells.sort(key=lambda x: -x["shares"])

    # 3) 持仓监控
    positions = []
    for i, t in enumerate(tics):
        p = acc["positions"].get(t, {})
        if p.get("shares", 0) <= 0:
            continue
        cost = p.get("cost", 0)
        pnl = round((closes[i] / cost - 1) * 100, 2) if cost > 0 else None
        sig = signal_of(float(mu[i]))
        sell_now = 0
        if a[i] < 0:
            sell_now = min(int(abs(a[i]) * HMAX), p.get("sellable", 0))
        positions.append({"tic": t, "shares": p.get("shares", 0),
                          "sellable": p.get("sellable", 0),
                          "cost": round(cost, 2) if cost else None,
                          "price": round(closes[i], 2),
                          "pnl_pct": pnl, "score": round(float(mu[i]), 3),
                          "signal": sig, "sell_now": sell_now})
    positions.sort(key=lambda x: -(x["pnl_pct"] if x["pnl_pct"] is not None else -999))

    out = {"date": str(rt_date or last_date), "mode": mode,
           "model": ("集成×%d" % n_models) if n_models > 1 else os.path.basename(args.model),
           "cash": round(acc["cash"], 2),
           "capital": args.capital,
           "ranking": ranking,
           "buy_orders": buys[:8], "sell_orders": sells[:8],
           "buy_count": len(buys), "sell_count": len(sells),
           "positions": positions}

    # 5) 市场环境仪表盘（给人工择时参考）
    ctx = {}
    for k in ["idx_sh_mom20", "idx_turnover", "idx_breadth", "idx_mf_net",
              "idx_vol20", "idx_sh_close"]:
        if k in day.columns:
            ctx[k] = round(float(day[k].iloc[0]), 2)
    out["context"] = ctx

    # 4) 自然语言解读（LLM 优先，规则兜底；--no-llm 时跳过）
    if args.no_llm:
        out["interpret"] = ""
        out["interpret_by"] = "none"
    else:
        try:
            summary = {
                "date": str(rt_date or last_date),
                "capital": args.capital,
                "ranking_top5": [{"name": r.get("name", ""), "tic": r["tic"],
                                  "score": r["score"], "chg_today": r.get("chg"),
                                  "mom20": r.get("mom20"), "news": r.get("sent"),
                                  "risk": r.get("risk"), "close": r.get("close")}
                                 for r in ranking[:5]],
                "buys": [{"name": b.get("name", ""), "tic": b["tic"], "shares": b["shares"],
                          "price": b["price"]} for b in buys[:5]],
                "sells": [{"name": s.get("name", ""), "tic": s["tic"], "shares": s["shares"],
                           "price": s["price"]} for s in sells[:5]],
                "positions": [{"name": p.get("name", ""), "tic": p["tic"], "shares": p["shares"],
                               "pnl_pct": p["pnl_pct"], "signal": p["signal"]}
                              for p in positions[:10]],
            }
            interp = llm_interpret(summary)
            out["interpret"] = interp if interp else rule_interpret(
                ranking, buys, sells, positions, out["date"], args.capital)
            out["interpret_by"] = "deepseek" if interp else "rule"
        except Exception:
            out["interpret"] = rule_interpret(ranking, buys, sells, positions,
                                              out["date"], args.capital)
            out["interpret_by"] = "rule"

    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print("=== %s (%s) ===" % (out["date"], mode))
        print("--- 期望收益排行 ---")
        for r in ranking:
            print("  %s 评分 %+.3f 动量 %s%% 现价 %.2f" % (r["tic"], r["score"], r["mom20"], r["close"]))
        print("--- 可执行建议 ---")
        print("买入:", ", ".join("%s %d股" % (b["tic"], b["shares"]) for b in buys[:8]) or "无")
        print("卖出:", ", ".join("%s %d股" % (s["tic"], s["shares"]) for s in sells[:8]) or "无")
        print("--- 持仓监控 ---")
        for p in positions:
            print("  %s %d股 成本%s 现价%.2f 盈亏%s%% 信号[%s] 建议卖出%d股" %
                  (p["tic"], p["shares"], p["cost"], p["price"], p["pnl_pct"], p["signal"], p["sell_now"]))


if __name__ == "__main__":
    main()

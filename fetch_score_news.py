# -*- coding: utf-8 -*-
"""
fetch_score_news.py v2 —— 新闻 + DeepSeek 打分管线（支持美股 / A股双模式）

流程:
1. 从新浪财经频道抓最近 N 天新闻标题（支持翻页）
2. 分批调用 DeepSeek API: 识别相关股票 + 推荐分1-5 + 风险分1-5
3. 按 (日期, 股票) 聚合写回对应 CSV 的 llm_sentiment / llm_risk 列

用法:
  ~/finrl_env/bin/python -u fetch_score_news.py --mode us      [--days 30]
  ~/finrl_env/bin/python -u fetch_score_news.py --mode ashare  [--days 30]
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

KEY_PATH = os.path.expanduser("~/FinRL_DeepSeek/.deepseek_key")
BASE = os.path.expanduser("~/FinRL_DeepSeek")

US_TICKERS = ['AAPL','ADBE','ADI','ADP','ADSK','AEP','ALGN','AMAT','AMD','AMGN','AMZN','ANSS','ASML',
'AVGO','AZN','BIIB','BKNG','BKR','CDNS','CHTR','CMCSA','COST','CPRT','CSCO','CSGP','CSX','CTAS',
'CTSH','DLTR','DXCM','EA','EBAY','ENPH','EXC','FANG','FAST','FTNT','GILD','GOOG','GOOGL','HON',
'IDXX','ILMN','INTC','INTU','ISRG','KDP','KLAC','LRCX','LULU','MAR','MCHP','MDLZ','MELI','META',
'MNST','MRVL','MSFT','MU','NFLX','NVDA','NXPI','ODFL','ON','ORLY','PANW','PAYX','PCAR','PEP',
'QCOM','REGN','ROST','SBUX','SIRI','SNPS','TMUS','TSLA','TXN','VRSK','VRTX','WBA','WBD','WDAY','XEL']

# A股: 代码+名称（沪深300市值前50）
ASHARE = [("601398","工商银行"),("601939","建设银行"),("601288","农业银行"),("601988","中国银行"),
("600941","中国移动"),("601857","中国石油"),("600938","中国海油"),("600519","贵州茅台"),
("300750","宁德时代"),("601138","工业富联"),("600036","招商银行"),("601628","中国人寿"),
("601088","中国神华"),("601318","中国平安"),("688981","中芯国际"),("300308","中际旭创"),
("601899","紫金矿业"),("002594","比亚迪"),("600900","长江电力"),("601328","交通银行"),
("601658","邮储银行"),("600028","中国石化"),("688256","寒武纪"),("000333","美的集团"),
("601728","中国电信"),("688041","海光信息"),("603259","药明康德"),("601998","中信银行"),
("002371","北方华创"),("600030","中信证券"),("002475","立讯精密"),("601166","兴业银行"),
("603993","洛阳钼业"),("601319","中国人保"),("600000","浦发银行"),("600183","生益科技"),
("600276","恒瑞医药"),("688012","中微公司"),("601601","中国太保"),("002415","海康威视"),
("601211","国泰海通"),("600150","中国船舶"),("300059","东方财富"),("000858","五粮液"),
("002916","深南电路"),("601225","陕西煤业"),("601919","中远海控"),("603986","兆易创新"),
("688008","澜起科技"),("002714","牧原股份")]


def make_system_prompt(mode):
    if mode == "ashare":
        stock_list = "; ".join("%s=%s" % (code, name) for code, name in ASHARE)
        intro = """你是资深A股分析师。你会收到一批财经新闻标题（中文，来自新浪财经）。
对每一条新闻完成三件事：
1. 判断它涉及下列 50 只 A 股中的哪些（输出其 6 位代码，可多个，可没有）。注意常用简称：茅台=600519、宁德=300750、比亚迪=002594、中芯=688981、平安=601318、招行=600036、工行=601398、美的=000333、寒武纪=688256、东方财富=300059。只从给定列表选。
2. 推荐分 sentiment：1=负面, 2=略负面, 3=中性, 4=略正面, 5=正面（对该股股价的含义）。
3. 风险分 risk：1=风险极低, 2=低, 3=中(无明确风险信号时默认), 4=高, 5=极高。"""
    else:
        stock_list = ", ".join(US_TICKERS)
        intro = """你是资深金融分析师。你会收到一批财经新闻标题（中文或英文，来自美股频道）。
对每一条新闻完成三件事：
1. 判断它涉及下列 84 只纳斯达克股票中的哪些（用 ticker，可多个，可没有）。注意中文名（苹果=AAPL、英伟达=NVDA、特斯拉=TSLA、微软=MSFT、谷歌=GOOGL、亚马逊=AMZN、脸书/Meta=META、奈飞=NFLX、英特尔=INTC、AMD=AMD、博通=AVGO、高通=QCOM、美光=MU、星巴克=SBUX、开市客=COST、百事=PEP、亿滋=MDLZ）。只从给定列表选。
2. 推荐分 sentiment：1=负面, 2=略负面, 3=中性, 4=略正面, 5=正面。
3. 风险分 risk：1=风险极低, 2=低, 3=中(无明确风险信号时默认), 4=高, 5=极高。"""
    return (intro + "\n\n股票列表: " + stock_list + """

严格输出 JSON 对象，格式:
{"results": [{"tickers": ["600519"], "sentiment": 3, "risk": 3}, ...]}
results 数组长度必须与输入的新闻条数完全一致。""")


def log(msg):
    print(msg, flush=True)


def fetch_news(days, lids):
    cutoff = time.time() - days * 86400
    seen, news = set(), []
    for lid in lids:
        page = 1
        while True:
            url = ("https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=%d"
                   "&k=&num=50&page=%d" % (lid, page))
            try:
                r = subprocess.run(["curl", "-s", "--max-time", "15", url],
                                   capture_output=True, text=True, timeout=20)
                j = json.loads(r.stdout)
            except Exception as e:
                log("lid %d 第 %d 页失败: %s" % (lid, page, e))
                break
            items = j.get("result", {}).get("data", [])
            if not items:
                break
            oldest = min(int(it.get("ctime", 0)) for it in items)
            for it in items:
                docid = it.get("docid")
                title = (it.get("title") or "").strip()
                if docid in seen or not title:
                    continue
                seen.add(docid)
                d = datetime.fromtimestamp(int(it.get("ctime", 0)), tz=timezone.utc)
                d = d.astimezone(timezone(timedelta(hours=8))).date().isoformat()  # 北京时间
                news.append({"date": d, "title": title})
            if oldest < cutoff or page > 200:
                break
            page += 1
            time.sleep(0.15)
    by_date = {}
    for n in news:
        by_date.setdefault(n["date"], []).append(n)
    news = [n for d in sorted(by_date, reverse=True) for n in by_date[d][:150]]
    log("抓取新闻 %d 条, 覆盖 %d 个日期: %s -> %s" %
        (len(news), len(by_date), min(by_date), max(by_date)))
    return news


def score_batch(api_key, sys_prompt, titles):
    user = "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(titles))
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    for attempt in range(3):
        try:
            r = requests.post("https://api.deepseek.com/chat/completions",
                              json=payload, timeout=90,
                              headers={"Authorization": "Bearer " + api_key})
            if r.status_code != 200:
                log("API %d: %s" % (r.status_code, r.text[:200]))
                time.sleep(3 + attempt * 3)
                continue
            content = r.json()["choices"][0]["message"]["content"]
            content = re.sub(r"^```json|```$", "", content.strip()).strip()
            data = json.loads(content)
            results = data.get("results", [])
            if len(results) == len(titles):
                return results
            log("结果数量不匹配: %d vs %d, 重试" % (len(results), len(titles)))
        except Exception as e:
            log("API 异常: %s" % e)
        time.sleep(3 + attempt * 3)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ashare",
                        choices=["us", "ashare", "ashare_idx2"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--force", action="store_true",
                        help="忽略已打分日期过滤(回填用)")
    args = parser.parse_args()

    if args.mode in ("ashare", "ashare_idx2"):
        lids = [2515, 2516]
        csv_path = os.path.join(BASE, "ashare_idx2_2022_2026.csv" if args.mode == "ashare_idx2"
                                else "ashare_2022_2026.csv")
        # 认全池股票: 一线/二线/三线 + 主池 的股票代码并集
        try:
            uni = set()
            for name in ["ashare_idx2_2022_2026.csv", "ashare_idx2_pool2_2022_2026.csv",
                         "ashare_idx2_cyb50_2022_2026.csv", "ashare_2022_2026.csv"]:
                d = pd.read_csv(os.path.join(BASE, name))
                uni.update(d["tic"].astype(str).str.zfill(6).unique())
            valid_tickers = sorted(uni)
            log("全池有效股票 %d 只" % len(valid_tickers))
        except Exception:
            valid_tickers = [c for c, _ in ASHARE]
    else:
        lids, csv_path = [2517], os.path.join(BASE, "recent_2023_2026.csv")
        valid_tickers = US_TICKERS

    with open(KEY_PATH) as f:
        api_key = f.read().strip()
    sys_prompt = make_system_prompt(args.mode)
    log("模式 %s, DeepSeek API key 已加载" % args.mode)

    news = fetch_news(args.days, lids)

    # 跳过非交易日（周末/节假日无行情行，打了分也没处写）和已打分日期（--force 忽略）
    try:
        existing = pd.read_csv(csv_path)
        trading_dates = set(existing["date"].astype(str).unique())
        scored = existing[existing["llm_sentiment"].astype(float) != 0]
        covered_dates = set(scored["date"].astype(str).unique())
        before = len(news)
        kept = [n for n in news if n["date"] in trading_dates]
        nontrading = before - len(kept)
        if not args.force:
            news = [n for n in kept if n["date"] not in covered_dates]
        else:
            news = kept
        log("跳过非交易日 %d 条 + 已打分日期 %d 个, 待打分 %d -> %d 条" %
            (nontrading, len(covered_dates), before, len(news)))
    except Exception as e:
        log("读取已有打分失败(继续全量打分): %s" % e)

    per = {}
    for i in range(0, len(news), args.batch):
        batch = news[i:i + args.batch]
        titles = [n["title"] for n in batch]
        results = score_batch(api_key, sys_prompt, titles)
        if results is None:
            log("批次 %d-%d 打分失败，跳过" % (i, i + len(batch)))
            continue
        for n, res in zip(batch, results):
            if args.mode == "ashare":
                tks = [str(t).zfill(6) for t in (res.get("tickers") or [])
                       if str(t).zfill(6) in valid_tickers]
            else:
                tks = [t.upper() for t in (res.get("tickers") or []) if t.upper() in valid_tickers]
            s, rk = res.get("sentiment"), res.get("risk")
            if not tks or not isinstance(s, int) or not isinstance(rk, int):
                continue
            s, rk = max(1, min(5, s)), max(1, min(5, rk))
            for t in set(tks):
                per.setdefault(n["date"], {}).setdefault(t, []).append((s, rk))
        log("已打分 %d/%d 条" % (min(i + args.batch, len(news)), len(news)))
        time.sleep(0.3)

    agg = {}
    for date, d in per.items():
        for tic, scores in d.items():
            if not scores:
                continue
            s = round(sum(x[0] for x in scores) / len(scores))
            rk = round(sum(x[1] for x in scores) / len(scores))
            agg[(date, tic)] = (max(1, min(5, s)), max(1, min(5, rk)))

    df = pd.read_csv(csv_path)
    # 防止反复读写累积多层索引列
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    updated = 0
    for (date, tic), (s, rk) in agg.items():
        if args.mode in ("ashare", "ashare_idx2"):
            mask = (df["date"].astype(str) == date) & (df["tic"].astype(str).str.zfill(6) == str(tic).zfill(6))
        else:
            mask = (df["date"].astype(str) == date) & (df["tic"] == tic)
        if mask.any():
            df.loc[mask, "llm_sentiment"] = s
            df.loc[mask, "llm_risk"] = rk
            updated += mask.sum()
    df.to_csv(csv_path, index=True)
    # ashare 家族: 一次打分同步写入所有池（主池/一线/二线/三线）
    if args.mode in ("ashare", "ashare_idx2"):
        for extra in ["ashare_2022_2026.csv", "ashare_idx2_2022_2026.csv",
                      "ashare_idx2_pool2_2022_2026.csv", "ashare_idx2_cyb50_2022_2026.csv"]:
            ep = os.path.join(BASE, extra)
            if ep == os.path.abspath(csv_path) or not os.path.exists(ep):
                continue
            try:
                df2 = pd.read_csv(ep)
                df2 = df2.loc[:, ~df2.columns.astype(str).str.startswith("Unnamed")]
                n2 = 0
                for (date, tic), (s, rk) in agg.items():
                    mask = (df2["date"].astype(str) == date) & (df2["tic"].astype(str).str.zfill(6) == str(tic).zfill(6))
                    if mask.any():
                        df2.loc[mask, "llm_sentiment"] = s
                        df2.loc[mask, "llm_risk"] = rk
                        n2 += mask.sum()
                df2.to_csv(ep, index=True)
                log("同步写入 %s %d 条" % (extra, n2))
            except Exception as e:
                log("%s 回写失败(忽略): %s" % (extra, e))
    log("已更新 %d 个 (日期,股票) 记录, 覆盖 %d 个股票, %d 个日期" %
        (updated, len(set(k[1] for k in agg)), len(set(k[0] for k in agg))))
    sample = df[df["llm_sentiment"].astype(float) != 0].tail(5)
    log("样例(近期含真实信号的记录):")
    log(sample[["date", "tic", "llm_sentiment", "llm_risk"]].to_string(index=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
make_ashare_data.py —— 下载 A 股（沪深300市值前50）2022-2026 日线（新浪源，国内直连），
处理成与训练管线完全相同的列结构。
注意: 环境沿用论文的美股模拟器，暂未建模 A 股 T+1/涨跌停/100股整手（后续可加）。
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from stockstats import StockDataFrame as Sdf

LOG_PATH = "/home/ddl/make_ashare_progress.log"
LOG = open(LOG_PATH, "w", encoding="utf-8")


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


_ap = argparse.ArgumentParser()
_ap.add_argument("--wide", action="store_true", help="多行业宽基池(~100只)")
_ap.add_argument("--idx", action="store_true", help="50池 + 大盘/板块指数特征")
_ap.add_argument("--idx2", action="store_true", help="50池 + 指数特征v2(成交额/主力资金/波动率/宽度)")
_ap.add_argument("--pool", type=str, default="top50",
                 choices=["top50", "second50", "cyb50"],
                 help="股票池: top50=沪深300前50 / second50=沪深300第51-100 / cyb50=创业板前50")
_args, _ = _ap.parse_known_args()

_BASE_NAME = "ashare"
if _args.wide:
    _BASE_NAME = "ashare_wide"
elif _args.idx:
    _BASE_NAME = "ashare_idx"
elif _args.idx2:
    _BASE_NAME = "ashare_idx2"
_POOL_SUFFIX = {"top50": "", "second50": "_pool2", "cyb50": "_cyb50"}[_args.pool]
OUT = "/home/ddl/FinRL_DeepSeek/%s%s_2022_2026.csv" % (_BASE_NAME, _POOL_SUFFIX)
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}

# 大盘指数 (新浪代码)
IDX_LIST = [("sh000001", "上证指数"), ("sh000300", "沪深300")]

# 板块分类规则（用于自算板块等权日收益）
SEC_RULES = [
    ("银行", lambda n: "银行" in n),
    ("非银", lambda n: any(k in n for k in ("保险", "证券", "财富", "海通", "平安"))),
    ("能源电力", lambda n: any(k in n for k in ("石油", "石化", "海油", "神华", "煤",
                                                  "电力", "核电", "广核", "长江", "三峡"))),
    ("科技", lambda n: any(k in n for k in ("中芯", "寒武", "海光", "中微", "澜起", "兆易",
                                             "北方华创", "中际", "工业富联", "生益", "深南",
                                             "立讯", "海康", "韦尔", "紫光"))),
    ("消费", lambda n: any(k in n for k in ("茅台", "五粮", "泸州", "汾酒", "洋河", "美的",
                                             "格力", "海尔", "伊利", "海天", "牧原", "新希望"))),
    ("医药", lambda n: any(k in n for k in ("医药", "药明", "恒瑞", "迈瑞", "爱尔"))),
]


def sec_of(name):
    for sec, rule in SEC_RULES:
        if rule(name):
            return sec
    return "其他"

# 多行业宽基池（保留原50只 + 按行业补齐），约100只
WIDE = [
("601398","工商银行"),("601939","建设银行"),("601288","农业银行"),("601988","中国银行"),
("600036","招商银行"),("601166","兴业银行"),("600000","浦发银行"),("601328","交通银行"),
("601658","邮储银行"),("601998","中信银行"),("000001","平安银行"),("002142","宁波银行"),
("601318","中国平安"),("601628","中国人寿"),("601601","中国太保"),("601319","中国人保"),
("600030","中信证券"),("601688","华泰证券"),("600999","招商证券"),("300059","东方财富"),("601211","国泰海通"),
("600519","贵州茅台"),("000858","五粮液"),("000568","泸州老窖"),("600809","山西汾酒"),("002304","洋河股份"),
("600887","伊利股份"),("603288","海天味业"),("000333","美的集团"),("000651","格力电器"),("600690","海尔智家"),
("600028","中国石化"),("601857","中国石油"),("600938","中国海油"),("601088","中国神华"),("601225","陕西煤业"),("601898","中煤能源"),
("600900","长江电力"),("601985","中国核电"),("600905","三峡能源"),("601012","隆基绿能"),("300274","阳光电源"),("600795","国电电力"),
("688981","中芯国际"),("688256","寒武纪"),("688041","海光信息"),("688012","中微公司"),("688008","澜起科技"),
("603986","兆易创新"),("002371","北方华创"),("002049","紫光国微"),("603501","韦尔股份"),("600584","长电科技"),("603160","汇顶科技"),
("002475","立讯精密"),("002415","海康威视"),("002241","歌尔股份"),("000725","京东方A"),("601138","工业富联"),("300308","中际旭创"),("002916","深南电路"),("600183","生益科技"),
("002594","比亚迪"),("600104","上汽集团"),("601633","长城汽车"),("601127","赛力斯"),("300750","宁德时代"),("300014","亿纬锂能"),("002460","赣锋锂业"),
("600276","恒瑞医药"),("603259","药明康德"),("300760","迈瑞医疗"),("300015","爱尔眼科"),("600436","片仔癀"),("000538","云南白药"),
("601899","紫金矿业"),("603993","洛阳钼业"),("600362","江西铜业"),("600309","万华化学"),("002466","天齐锂业"),
("600941","中国移动"),("601728","中国电信"),("600050","中国联通"),("688111","金山办公"),("002230","科大讯飞"),
("601668","中国建筑"),("601390","中国中铁"),("600048","保利发展"),("600585","海螺水泥"),
("600150","中国船舶"),("601989","中国重工"),("600760","中航沈飞"),
("601919","中远海控"),("601006","大秦铁路"),("002352","顺丰控股"),
("002714","牧原股份"),("000876","新希望"),
]


def sina_symbol(code):
    code = str(code).zfill(6)
    return ("sh" if code[0] in "69" else "sz") + code


def get_node(node, pages=4):
    """新浪板块成分股（node: hs300/cyb 等），返回按市值降序；带重试与断页"""
    out = []
    for page in range(1, pages + 1):
        url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               "Market_Center.getHQNodeData?page=%d&num=80&sort=symbol&asc=1&node=%s"
               % (page, node))
        arr = None
        for _ in range(3):
            try:
                r = requests.get(url, timeout=20, headers=HEADERS)
                arr = r.json()
                break
            except Exception:
                time.sleep(1.0)
        if not arr:
            break  # 空页 = 已到末尾
        out += arr
        time.sleep(0.3)
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("mktcap", ascending=False)
    return df


def get_hs300():
    """沪深300成分股，返回按市值降序"""
    return get_node("hs300")


def get_kline(symbol, datalen=1023, retries=3):
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%%20_=/"
           "CN_MarketDataService.getKLineData?symbol=%s&scale=240&ma=no&datalen=%d"
           % (symbol, datalen))
    r = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            break
        except Exception:
            time.sleep(1.0)
    if r is None:
        return None
    t = r.text
    i, j = t.find("(["), t.rfind("])")
    if i == -1 or j == -1:
        return None
    arr = json.loads(t[i + 1:j + 1])
    df = pd.DataFrame(arr)
    if df.empty:
        return None
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_moneyflow(sym):
    """新浪主力资金流向历史（逐日净流入）"""
    out = []
    for page in range(1, 12):
        url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               "MoneyFlow.ssl_qsfx_zjlrqs?page=%d&num=100&sort=opendate&asc=0&daima=%s"
               % (page, sym))
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            arr = r.json()
        except Exception:
            break
        if not arr:
            break
        out += arr
        if len(arr) < 100:
            break
        time.sleep(0.2)
    if not out:
        return None
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["opendate"])
    for c in ["netamount", "cate_na"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "netamount", "cate_na"]].sort_values("date")


if _args.wide:
    uni = [(sina_symbol(c), c, n) for c, n in WIDE]
    log("宽基池 %d 只(多行业)" % len(uni))
elif _args.pool == "second50":
    hs = get_hs300()
    sel = hs.head(100).tail(50)
    uni = [(row["symbol"], row["code"], row["name"]) for _, row in sel.iterrows()]
    log("沪深300第51-100名(按市值): %d 只" % len(uni))
elif _args.pool == "cyb50":
    cyb = get_node("cyb", pages=20)
    sel = cyb.head(80)  # 多取30只备选, 后面剔除次新股
    uni = [(row["symbol"], row["code"], row["name"]) for _, row in sel.iterrows()]
    log("创业板候选(按市值): %d 只" % len(uni))
else:
    hs = get_hs300()
    top50 = hs.head(50)
    uni = [(row["symbol"], row["code"], row["name"]) for _, row in top50.iterrows()]
    log("沪深300前50(按市值): " + ", ".join("%s(%s)" % (c, n) for _, c, n in uni))

frames = {}
for sym, code, name in uni:
    try:
        df = get_kline(sym)
        if df is None or df.empty:
            log("%s %s 无数据" % (code, name)); continue
        if _args.pool == "cyb50" and len(df) < 900:
            log("%s %s 历史不足900天(2023年后上市), 剔除" % (code, name)); continue
        df["tic"] = code
        frames[code] = df
        log("%s %s: %d 天, %s -> %s" % (code, name, len(df),
            df["date"].min().date(), df["date"].max().date()))
    except Exception as e:
        log("%s %s 失败: %s" % (code, name, e))
    time.sleep(0.3)

if _args.pool == "cyb50" and len(frames) > 50:
    frames = {c: f for c, f in list(frames.items())[:50]}  # 按市值序取前50
    log("创业板取前50只(已剔次新股)")

# 用「并集」对齐交易日 + 前向填充：
# 个别股票停牌/数据滞后只冻结它自己的价格，不再拖累全市场日期
all_dates = sorted(set().union(*[set(f["date"]) for f in frames.values()]))
date_idx = pd.DatetimeIndex(all_dates)
panel, keep = [], []
for tic, f in frames.items():
    d = f.set_index("date").reindex(date_idx)
    d = d.ffill()  # 先前向填充（停牌/末日缺失都冻结价格）
    if d["close"].isna().any():
        d = d.dropna(subset=["close"])  # 只删开头缺失（历史起点晚）
        if d.empty:
            log("无数据剔除: " + tic); continue
    d["volume"] = d["volume"].fillna(0)
    d = d.reset_index().rename(columns={"index": "date"})
    panel.append(d)
    keep.append(tic)

df = pd.concat(panel, ignore_index=True)
# 只保留「全部股票都在场」的日期（环境要求每天恰好 stock_dim 行）
cnt = df.groupby(df["date"].astype(str)).size()
ok_dates = set(cnt[cnt == len(keep)].index)
n_dropped = len(cnt) - len(ok_dates)
df = df[df["date"].astype(str).isin(ok_dates)]
log("\n保留 %d 只, %s -> %s, 行数 %d (剔除未满员日期 %d 个)" % (len(keep),
    df.date.min().date(), df.date.max().date(), len(df), n_dropped))

if _args.idx or _args.idx2:
    log("计算大盘/板块指数特征...")
    idx_frames = {}
    idx_kframes = {}
    for sym, name in IDX_LIST:
        try:
            k = get_kline(sym)
        except Exception:
            k = None
        if k is None:
            log("指数 %s 无数据" % name)
            continue
        idx_frames[sym] = k.set_index("date")["close"]
        idx_kframes[sym] = k.set_index("date")
        log("指数 %s: %d 天" % (name, len(k)))
    idx_df = pd.DataFrame(index=pd.DatetimeIndex(sorted(all_dates)))
    sh = idx_frames.get("sh000001")
    hs = idx_frames.get("sh000300")
    if sh is not None:
        idx_df["idx_sh_close"] = sh.reindex(idx_df.index).ffill()
        idx_df["idx_sh_mom5"] = (idx_df["idx_sh_close"] / idx_df["idx_sh_close"].shift(5) - 1) * 100
        idx_df["idx_sh_mom20"] = (idx_df["idx_sh_close"] / idx_df["idx_sh_close"].shift(20) - 1) * 100
    if hs is not None:
        idx_df["idx_hs_close"] = hs.reindex(idx_df.index).ffill()
        idx_df["idx_hs_mom20"] = (idx_df["idx_hs_close"] / idx_df["idx_hs_close"].shift(20) - 1) * 100
    # 板块等权日收益（池内股票自算）
    name_map = {c: n for _, c, n in uni}
    g = df.copy()
    g["ret"] = g.groupby("tic")["close"].pct_change()
    g["sec"] = g["tic"].map(lambda c: sec_of(name_map.get(str(c).zfill(6), "")))
    sec_ret = g.groupby(["date", "sec"])["ret"].mean().unstack()
    for sec in ["银行", "非银", "能源电力", "科技", "消费", "医药", "其他"]:
        if sec in sec_ret.columns:
            idx_df["idx_sec_" + sec] = sec_ret[sec].reindex(idx_df.index) * 100
        else:
            idx_df["idx_sec_" + sec] = 0.0  # 池内无该板块: 固定置0, 保证列结构与训练数据一致

    if _args.idx2:
        log("计算 v2 特征: 成交额/波动率/宽度/主力资金...")
        # 两市成交额(亿元) + 动量
        if "sh000001" in idx_kframes:
            turn = idx_kframes["sh000001"]["volume"] / 1e8
            idx_df["idx_turnover"] = turn.reindex(idx_df.index).ffill()
            idx_df["idx_turn_mom5"] = (idx_df["idx_turnover"] / idx_df["idx_turnover"].shift(5) - 1) * 100
            idx_df["idx_turn_mom20"] = (idx_df["idx_turnover"] / idx_df["idx_turnover"].shift(20) - 1) * 100
        # 上证 20 日已实现波动率(%)
        if sh is not None:
            idx_df["idx_vol20"] = (idx_df["idx_sh_close"].pct_change().rolling(20).std() * 100
                                   ).reindex(idx_df.index)
        # 市场宽度: 池内上涨家数占比(%)
        up = (g["ret"] > 0).groupby(g["date"]).mean() * 100
        idx_df["idx_breadth"] = up.reindex(idx_df.index)
        # 主力资金: 池内 50 只净流入合计(亿元) 与 净流入/成交额比
        mf_frames = {}
        for _, code, name in uni:
            if code not in frames:
                continue  # 已被剔除的候选
            sym = sina_symbol(code)
            mf = get_moneyflow(sym)
            if mf is not None:
                mf_frames[code] = mf
            log("资金流 %s %s: %s" % (code, name,
                ("%d 天" % len(mf)) if mf is not None else "无数据"))
        if mf_frames:
            mf_all = pd.concat(mf_frames.values())
            mf_agg = mf_all.groupby("date").agg(net=("netamount", "sum"),
                                                vol=("cate_na", "sum"))
            idx_df["idx_mf_net"] = (mf_agg["net"] / 1e8).reindex(idx_df.index).ffill()
            idx_df["idx_mf_ratio"] = (mf_agg["net"] / mf_agg["vol"] * 100
                                      ).reindex(idx_df.index).ffill()

    idx_df = idx_df.reset_index().rename(columns={"index": "date"})
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    df = df.merge(idx_df, on="date", how="left")
    log("指数特征列: " + ", ".join(c for c in idx_df.columns if c != "date"))

log("计算技术指标...")
df = df.sort_values(["tic", "date"]).reset_index(drop=True)
stock = Sdf.retype(df.copy())
for col in ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30"]:
    out = []
    for tic in keep:
        s = stock[stock["tic"] == tic]
        out.append(pd.DataFrame({"tic": tic,
                                 "date": df[df["tic"] == tic]["date"].values,
                                 col: s[col].values}))
    df = df.merge(pd.concat(out, ignore_index=True), on=["tic", "date"], how="left")

sma = []
for tic in keep:
    g = df[df["tic"] == tic].sort_values("date")
    g = g.assign(close_30_sma=g["close"].rolling(30).mean(),
                 close_60_sma=g["close"].rolling(60).mean())
    sma.append(g[["tic", "date", "close_30_sma", "close_60_sma"]])
df = df.merge(pd.concat(sma, ignore_index=True), on=["tic", "date"], how="left")
df = df.sort_values(["date", "tic"])

df["day"] = pd.to_datetime(df["date"]).dt.weekday
df["turbulence"] = 0.0
df["llm_sentiment"] = 0.0
df["llm_risk"] = 3.0
# 保留已有的 DeepSeek 新闻打分（刷新行情不清零）
if os.path.exists(OUT):
    try:
        old = pd.read_csv(OUT, usecols=["date", "tic", "llm_sentiment", "llm_risk"])
        m = dict(zip(old["date"].astype(str) + "|" + old["tic"].astype(str).str.zfill(6),
                     zip(old["llm_sentiment"], old["llm_risk"])))
        key = df["date"].astype(str) + "|" + df["tic"].astype(str).str.zfill(6)
        vals = [m.get(k) for k in key]
        df["llm_sentiment"] = [v[0] if v else 0.0 for v in vals]
        df["llm_risk"] = [v[1] if v else 3.0 for v in vals]
        log("已保留 %d 条既有新闻打分" % sum(1 for v in vals if v))
    except Exception as e:
        log("保留新闻打分失败(忽略): %s" % e)

cols = ["date", "tic", "close", "high", "low", "open", "volume", "day",
        "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
        "close_30_sma", "close_60_sma", "turbulence",
        "llm_sentiment", "llm_risk"]
if _args.idx or _args.idx2:
    cols += [c for c in df.columns if str(c).startswith("idx_")]
df = df[cols].sort_values(["date", "tic"]).reset_index(drop=True)
before = len(df)
df = df.dropna(subset=["close_60_sma"])
log("预热期剔除: %d -> %d 行, 起始 %s" % (before, len(df), df.date.min().date()))
df.to_csv(OUT, index=True)
log("已保存: " + OUT)
log("缺失值: %s" % str(df.isna().sum()[df.isna().sum() > 0].to_dict()))
log("完成")

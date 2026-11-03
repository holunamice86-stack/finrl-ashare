# -*- coding: utf-8 -*-
"""names_cache.py —— 从新浪实时接口抓取股票中文名，缓存到 stock_names.json"""
import json
import os

import pandas as pd
import requests

BASE = os.path.expanduser("~/FinRL_DeepSeek")
CSV = os.path.join(BASE, "ashare_2022_2026.csv")
CSV_WIDE = os.path.join(BASE, "ashare_wide_2022_2026.csv")
OUT = os.path.join(BASE, "stock_names.json")
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}


def sina_symbol(code):
    code = str(code).zfill(6)
    return ("sh" if code[0] in "69" else "sz") + code


def main():
    tics = set()
    for name in ["ashare_2022_2026.csv", "ashare_wide_2022_2026.csv",
                 "ashare_idx2_2022_2026.csv", "ashare_idx2_pool2_2022_2026.csv",
                 "ashare_idx2_cyb50_2022_2026.csv"]:
        try:
            d = pd.read_csv(os.path.join(BASE, name))
            tics.update(d["tic"].astype(str).str.zfill(6).unique())
        except Exception:
            pass
    tics = sorted(tics)
    names = {}
    url = "https://hq.sinajs.cn/list=" + ",".join(sina_symbol(t) for t in tics)
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.encoding = "gbk"
    for line in r.text.strip().splitlines():
        if '="' not in line:
            continue
        head, body = line.split('="', 1)
        sym = head.split("_")[-1].strip()
        code = sym[2:]
        f = body.rstrip('";').split(",")
        if f and f[0]:
            names[code] = f[0].strip()
    missing = [t for t in tics if t not in names]
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(names, fp, ensure_ascii=False, indent=1)
    print("已保存 %d 个股票名到 %s" % (len(names), OUT))
    if missing:
        print("未取到名字:", missing)


if __name__ == "__main__":
    main()

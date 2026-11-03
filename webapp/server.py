# -*- coding: utf-8 -*-
"""FinRL-DeepSeek 本地 Web 操作台后端 (Flask)"""
import collections
import glob
import json
import os
import secrets
import subprocess
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "FinRL_DeepSeek")
WEB = os.path.join(BASE, "webapp")
STATIC = os.path.join(WEB, "static")

# ---- 访问密钥（公网穿透时保护操作台） ----
WEB_KEY = ""
try:
    with open(os.path.join(BASE, ".web_key"), encoding="utf-8") as f:
        WEB_KEY = f.read().strip()
except Exception:
    pass


def _safe(o):
    """把 NaN/Infinity 转成 None（浏览器 JSON 解析不接受 NaN）"""
    import math
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_safe(v) for v in o]
    return o
LOG_DIR = os.path.join(WEB, "logs")
PY = os.path.join(HOME, "finrl_env", "bin", "python")

os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")


@app.before_request
def _auth():
    if not WEB_KEY or request.path.startswith("/static"):
        return None
    if request.args.get("key") == WEB_KEY:
        request._webkey_ok = True  # 有效密钥: 记下, after_request 种 cookie
        return None
    if request.cookies.get("webkey") == WEB_KEY:
        return None
    if request.path in ("/", "/index.html", "/favicon.ico"):
        return None  # 首页永远可打开, 前端会引导输入密钥
    return jsonify({"ok": False,
                    "msg": "需要访问密钥。请在网址后加 ?key=你的密钥（首次进入后会自动记住）"}), 401


@app.after_request
def _set_cookie(resp):
    if getattr(request, "_webkey_ok", False):
        resp.set_cookie("webkey", WEB_KEY, max_age=60 * 60 * 24 * 30)
    return resp

JOB = {"proc": None, "log": collections.deque(maxlen=300),
       "logfile": None, "name": "", "started": ""}

_STATUS_CACHE = {"t": 0.0, "payload": ""}  # /api/status 节流缓存，防止坏页面刷爆


def base_env():
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONPATH"] = os.path.join(HOME, "spinningup")
    return env


# ---- 股票中文名 ----
NAMES_FILE = os.path.join(BASE, "stock_names.json")


def load_names():
    try:
        with open(NAMES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def with_names(items):
    """给 dict 列表加 name 字段（按 tic 匹配）"""
    names = load_names()
    for x in (items or []):
        t = str(x.get("tic", "")).zfill(6)
        if t in names:
            x["name"] = names[t]
    return items


def prod_model():
    """生产模型: v2 指数特征模型优先; 其次老冠军; 都没有返回 None"""
    for name in ("ashare_idx2_30ep_best.pth", "ashare_30ep_v2_best.pth"):
        prod = os.path.join(BASE, "trained_models", name)
        if os.path.exists(prod):
            return os.path.join("trained_models", name)
    bests = sorted(glob.glob(os.path.join(BASE, "trained_models", "*_best.pth")),
                   key=os.path.getmtime, reverse=True)
    if bests:
        return os.path.join("trained_models", os.path.basename(bests[0]))
    return None


def _reader(proc):
    for line in proc.stdout:
        JOB["log"].append(line.rstrip())
        if JOB["logfile"]:
            JOB["logfile"].write(line)
            JOB["logfile"].flush()
    if JOB["logfile"]:
        JOB["logfile"].close()
        JOB["logfile"] = None


# ---- DeepSeek 新闻信号任务 ----
NEWS = {"proc": None, "log": collections.deque(maxlen=100)}
NEWS_INFO_CACHE = {"t": 0.0, "payload": None}
DATA = {"proc": None, "log": collections.deque(maxlen=100)}


def _news_reader(proc):
    for line in proc.stdout:
        NEWS["log"].append(line.rstrip())


def _data_reader(proc):
    for line in proc.stdout:
        DATA["log"].append(line.rstrip())


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/api/status")
def status():
    now = time.time()
    if now - _STATUS_CACHE["t"] < 0.5:
        return _STATUS_CACHE["payload"]  # 0.5 秒内重复请求直接返回缓存
    running = JOB["proc"] is not None and JOB["proc"].poll() is None
    news_running = NEWS["proc"] is not None and NEWS["proc"].poll() is None
    data_running = DATA["proc"] is not None and DATA["proc"].poll() is None
    payload = jsonify({
        "running": running,
        "name": JOB["name"],
        "started": JOB["started"],
        "log": list(JOB["log"])[-200:],
        "news_running": news_running,
        "news_log": list(NEWS["log"])[-30:],
        "data_running": data_running,
        "data_log": list(DATA["log"])[-10:],
    })
    _STATUS_CACHE["t"] = now
    _STATUS_CACHE["payload"] = payload
    return payload


@app.route("/api/news-info")
def news_info():
    now = time.time()
    if NEWS_INFO_CACHE["payload"] is not None and now - NEWS_INFO_CACHE["t"] < 60:
        return NEWS_INFO_CACHE["payload"]
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join(BASE, "ashare_2022_2026.csv"),
                         usecols=["date", "tic", "llm_sentiment"])
        sig = df[df["llm_sentiment"].astype(float) != 0]
        payload = jsonify({
            "covered_rows": int(len(sig)),
            "covered_dates": sorted(sig["date"].astype(str).unique().tolist()),
            "covered_stocks": sorted(sig["tic"].astype(str).str.zfill(6).unique().tolist()),
            "csv_last_date": str(df["date"].max()),
        })
    except Exception as e:
        payload = jsonify({"error": str(e)})
    NEWS_INFO_CACHE["t"] = now
    NEWS_INFO_CACHE["payload"] = payload
    return payload


@app.route("/api/news-refresh", methods=["POST"])
def news_refresh():
    if NEWS["proc"] is not None and NEWS["proc"].poll() is None:
        return jsonify({"ok": False, "msg": "新闻刷新已在运行中"}), 409
    data = request.get_json(force=True)
    days = int(data.get("days", 30))
    mode = str(data.get("mode") or "ashare_idx2")
    proc = subprocess.Popen([PY, "fetch_score_news.py", "--mode", mode,
                             "--days", str(days)],
                            cwd=BASE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=base_env())
    NEWS.update({"proc": proc, "log": collections.deque(maxlen=100)})
    threading.Thread(target=_news_reader, args=(proc,), daemon=True).start()
    return jsonify({"ok": True, "msg": "新闻刷新已启动（抓取+DeepSeek打分，约 3-8 分钟）"})


@app.route("/api/signal-demo", methods=["POST"])
def signal_demo():
    try:
        # 系统已聚焦 A 股，固定使用 A 股数据
        csv_path = "ashare_2022_2026.csv"
        # 找该数据源最新训练的模型
        model, hidden = None, [512, 512]
        for f in sorted(glob.glob(os.path.join(BASE, "trained_models", "*.pth")),
                        key=os.path.getmtime, reverse=True):
            stem = os.path.basename(f)[:-4]
            mp = os.path.join(BASE, "trained_models", stem + ".json")
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as mf:
                    meta = json.load(mf)
                if (meta.get("data_csv") or "") == csv_path:
                    model = os.path.join("trained_models", os.path.basename(f))
                    hidden = [int(meta.get("hid", 512))] * int(meta.get("layers", 2))
                    break
        if model is None:
            return jsonify({"ok": False, "msg": "该数据源还没有训练好的模型，请先训练"}), 409
        cmd = [PY, "signal_demo.py", "--json", "--csv", csv_path,
               "--model", model, "--hidden"] + [str(h) for h in hidden]
        r = subprocess.run(cmd, cwd=BASE, env=base_env(),
                           capture_output=True, text=True, timeout=300)
        lines = [x for x in r.stdout.strip().splitlines() if x.strip().startswith("{")]
        if not lines:
            return jsonify({"ok": False, "msg": "生成失败: " + r.stderr[-300:]}), 500
        data = json.loads(lines[-1])
        for key in ("buys", "sells"):
            with_names(data.get("with_news", {}).get(key))
            with_names(data.get("without_news", {}).get(key))
        with_names(data.get("news_stocks"))
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "msg": "生成失败: %s" % e}), 500


@app.route("/api/train", methods=["POST"])
def train():
    if JOB["proc"] is not None and JOB["proc"].poll() is None:
        return jsonify({"ok": False, "msg": "已有训练在运行，请先停止"}), 409
    data = request.get_json(force=True)
    epochs = int(data.get("epochs", 20))
    steps = int(data.get("steps", 20000))
    alpha = float(data.get("alpha", 0.85))
    beta = float(data.get("beta", 3000.0))
    cvar_clip = float(data.get("cvar_clip", 0.05))
    hid = int(data.get("hid", 512))
    layers = int(data.get("layers", 2))
    seed = int(data.get("seed", 0))
    model_name = os.path.basename(str(data.get("model_name") or "agent_cppo_deepseek_web"))
    model_name = model_name.replace(".pth", "").replace(".json", "")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, "train_%s_%s.log" % (ts, model_name))
    logfile = open(log_path, "w", encoding="utf-8")
    data_source = str(data.get("data_source") or "ashare")
    resume = str(data.get("resume") or "")
    cmd = [PY, "train_web.py",
           "--epochs", str(epochs), "--steps", str(steps),
           "--alpha", str(alpha), "--beta", str(beta),
           "--cvar-clip", str(cvar_clip),
           "--hid", str(hid), "--l", str(layers),
           "--seed", str(seed), "--model-name", model_name]
    if data_source == "ashare":
        # A股: 2022-10 -> 2025-06 训练段（2025-07 之后留给回测）
        cmd += ["--data-csv", "ashare_2022_2026.csv",
                "--data-start", "2022-10-01", "--data-end", "2025-06-30"]
    if data_source == "ashare_wide":
        # A股宽基(~100只多行业): 同样切法
        cmd += ["--data-csv", "ashare_wide_2022_2026.csv",
                "--data-start", "2022-10-01", "--data-end", "2025-06-30"]
    if data_source == "ashare_idx":
        # A股50池 + 大盘/板块指数特征
        cmd += ["--data-csv", "ashare_idx_2022_2026.csv",
                "--data-start", "2022-10-01", "--data-end", "2025-06-30"]
    if data_source == "ashare_idx2":
        # A股50池 + 指数特征v2(成交额/主力资金/波动率/宽度)
        cmd += ["--data-csv", "ashare_idx2_2022_2026.csv",
                "--data-start", "2022-10-01", "--data-end", "2025-06-30"]
    if resume:
        cmd += ["--resume", os.path.join("trained_models", os.path.basename(resume))]
    if data.get("data_start"):
        cmd += ["--data-start", str(data["data_start"])]
    if data.get("data_end"):
        cmd += ["--data-end", str(data["data_end"])]
    if data.get("pi_lr"):
        cmd += ["--pi-lr", str(float(data["pi_lr"]))]
    if data.get("save_freq"):
        cmd += ["--save-freq", str(int(data["save_freq"]))]
    proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=base_env())
    JOB.update({"proc": proc, "log": collections.deque(maxlen=300),
                "logfile": logfile, "name": model_name,
                "started": ts, "log_path": log_path})
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    return jsonify({"ok": True, "msg": "训练已启动: " + model_name})


@app.route("/api/stop", methods=["POST"])
def stop():
    if JOB["proc"] is None or JOB["proc"].poll() is not None:
        return jsonify({"ok": False, "msg": "没有正在运行的训练"}), 409
    JOB["proc"].terminate()
    try:
        JOB["proc"].wait(timeout=10)
    except subprocess.TimeoutExpired:
        JOB["proc"].kill()
    return jsonify({"ok": True, "msg": "已停止训练"})


@app.route("/api/models")
def models():
    files = sorted(glob.glob(os.path.join(BASE, "trained_models", "*.pth")))
    out = []
    for f in files:
        stem = os.path.basename(f)[:-4]
        meta_path = os.path.join(BASE, "trained_models", stem + ".json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as mf:
                meta = json.load(mf)
        out.append({
            "name": os.path.basename(f),
            "size_mb": round(os.path.getsize(f) / 1e6, 2),
            "meta": meta,
        })
    return jsonify(out)


@app.route("/api/backtest", methods=["POST"])
def backtest():
    data = request.get_json(force=True)
    model = os.path.basename(str(data.get("model") or ""))
    if not model.endswith(".pth"):
        return jsonify({"ok": False, "msg": "模型名非法"}), 400
    stem = model[:-4]
    outdir = os.path.join(STATIC, "results", stem)
    os.makedirs(outdir, exist_ok=True)

    hidden = [512, 512]
    meta_path = os.path.join(BASE, "trained_models", stem + ".json")
    extra = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        hidden = [int(meta.get("hid", 512))] * int(meta.get("layers", 2))
        if meta.get("data_csv"):
            # 本地数据训练的模型 -> 用 2025-07 之后的留出段回测
            extra = ["--data-csv", meta["data_csv"],
                     "--data-start", "2025-07-01", "--data-end", "2030-01-01"]
    elif stem.startswith("ashare"):
        # 无元信息的 A 股模型（如 *_best 检查点）也走 A 股留出段
        extra = ["--data-csv", "ashare_2022_2026.csv",
                 "--data-start", "2025-07-01", "--data-end", "2030-01-01"]

    cmd = [PY, "use_model.py", "--model", os.path.join("trained_models", model),
           "--outdir", outdir, "--hidden"] + [str(h) for h in hidden] + extra
    eval_mode = str(data.get("eval_mode") or "mean")
    if eval_mode == "sample":
        cmd += ["--eval-mode", "sample", "--eval-n", str(int(data.get("eval_n", 5)))]
    try:
        subprocess.run(cmd, cwd=BASE, env=base_env(), timeout=1800,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "msg": "回测超时(>30分钟)"}), 500
    res_path = os.path.join(outdir, "回测_结果.json")
    if not os.path.exists(res_path):
        return jsonify({"ok": False, "msg": "回测失败，请查看 webapp/server.log"}), 500
    with open(res_path, encoding="utf-8") as f:
        summary = json.load(f)
    return jsonify({"ok": True, "summary": summary,
                    "img": "/static/results/%s/回测_账户曲线.png" % stem,
                    "csv": "/static/results/%s/回测_账户明细.csv" % stem})


@app.route("/api/data-refresh", methods=["POST"])
def data_refresh():
    if DATA["proc"] is not None and DATA["proc"].poll() is None:
        return jsonify({"ok": False, "msg": "行情刷新已在运行中"}), 409
    mode = str(request.get_json(force=True).get("mode") or "ashare_idx2")
    if mode == "ashare_wide":
        cmd = "cd %s && %s make_ashare_data.py --wide && %s names_cache.py" % (BASE, PY, PY)
    elif mode == "ashare_idx":
        cmd = "cd %s && %s make_ashare_data.py --idx && %s names_cache.py" % (BASE, PY, PY)
    elif mode == "ashare_idx2":
        cmd = "cd %s && %s make_ashare_data.py --idx2 && %s names_cache.py" % (BASE, PY, PY)
    else:
        cmd = "cd %s && %s make_ashare_data.py && %s names_cache.py" % (BASE, PY, PY)
    proc = subprocess.Popen(["bash", "-c", cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=base_env())
    DATA.update({"proc": proc, "log": collections.deque(maxlen=100)})
    threading.Thread(target=_data_reader, args=(proc,), daemon=True).start()
    return jsonify({"ok": True,
                    "msg": ("A股行情" if mode == "ashare" else "美股近期行情") + "刷新已启动（约 1-2 分钟）"})


# ---- 建议/选股榜 后台任务（秒回+轮询, 防长连接被网络掐断） ----
ADV_JOB = {"running": False, "result": None, "error": None}
TIER_JOB = {"running": False, "done": 0, "total": 3, "tiers": [],
            "result": None, "error": None}

# 多模型投票集成名单: 生产模型 + 三种子复现模型（存在才加入）
ENSEMBLE_NAMES = ["ashare_idx2_30ep_best.pth",
                  "ashare_idx2_30ep_s1_best.pth",
                  "ashare_idx2_30ep_s2_best.pth",
                  "ashare_idx2_30ep_s3_best.pth"]


def ensemble_models():
    return [n for n in ENSEMBLE_NAMES
            if os.path.exists(os.path.join(BASE, "trained_models", n))]


def _run_advice_job(job, capital, tier_label=None, csv_name=None):
    try:
        mods = ensemble_models() or ([os.path.basename(prod_model())]
                                     if prod_model() else [])
        if not mods:
            raise RuntimeError("没有可用的最佳模型")
        csv = csv_name or "ashare_idx2_2022_2026.csv"
        cmd = [PY, "advice.py", "--json", "--csv", csv,
               "--models", ",".join(mods), "--hidden", "512", "512"]
        if tier_label:
            cmd += ["--no-llm"]
        if capital > 0:
            cmd += ["--capital", str(capital)]
        r = subprocess.run(cmd, cwd=BASE, env=base_env(), capture_output=True,
                           text=True, timeout=300)
        lines = [x for x in r.stdout.strip().splitlines() if x.strip().startswith("{")]
        if not lines:
            raise RuntimeError(r.stderr[-300:])
        data = json.loads(lines[-1])
        if tier_label:
            with_names(data.get("ranking"))
            with_names(data.get("buy_orders"))
            with_names(data.get("sell_orders"))
            job["result"] = {"label": tier_label, "data": data}
        else:
            for key in ("ranking", "buy_orders", "sell_orders", "positions"):
                with_names(data.get(key))
            job["result"] = {"ok": True, "data": data}
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["running"] = False


@app.route("/api/advice", methods=["POST"])
def advice():
    if ADV_JOB["running"]:
        return jsonify({"ok": False, "msg": "建议生成中, 请稍候"}), 409
    body = request.get_json(silent=True) or {}
    capital = float(body.get("capital") or 0)
    ADV_JOB.update(running=True, result=None, error=None)
    threading.Thread(target=_run_advice_job, args=(ADV_JOB, capital),
                     daemon=True).start()
    return jsonify({"ok": True, "msg": "生成中"})


@app.route("/api/advice-status")
def advice_status():
    return jsonify(_safe({"running": ADV_JOB["running"],
                          "result": ADV_JOB["result"], "error": ADV_JOB["error"]}))


@app.route("/api/advice-tiers", methods=["POST"])
def advice_tiers():
    """分层推荐: 一线/二线/三线（秒回+轮询）"""
    if TIER_JOB["running"]:
        return jsonify({"ok": False, "msg": "选股榜生成中, 请稍候"}), 409
    body = request.get_json(silent=True) or {}
    capital = float(body.get("capital") or 0)
    TIER_JOB.update(running=True, done=0, total=3, tiers=[], result=None,
                    error=None)

    def worker():
        tiers_def = [
            ("一线 · 沪深300龙头", "ashare_idx2_2022_2026.csv"),
            ("二线 · 沪深300中坚", "ashare_idx2_pool2_2022_2026.csv"),
            ("三线 · 创业板成长", "ashare_idx2_cyb50_2022_2026.csv"),
        ]
        try:
            for i, (label, csv_name) in enumerate(tiers_def):
                sub = {"running": True, "result": None, "error": None}
                _run_advice_job(sub, capital, tier_label=label, csv_name=csv_name)
                if sub["error"]:
                    raise RuntimeError("%s: %s" % (label, sub["error"]))
                TIER_JOB["tiers"] = TIER_JOB["tiers"] + [sub["result"]]
                TIER_JOB["done"] = i + 1
            TIER_JOB["result"] = {"ok": True, "tiers": TIER_JOB["tiers"]}
        except Exception as e:
            TIER_JOB["error"] = str(e)
        finally:
            TIER_JOB["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "msg": "生成中"})


@app.route("/api/advice-tiers-status")
def advice_tiers_status():
    return jsonify(_safe({"running": TIER_JOB["running"],
                          "done": TIER_JOB["done"], "total": TIER_JOB["total"],
                          "tiers": TIER_JOB["tiers"],
                          "result": TIER_JOB["result"], "error": TIER_JOB["error"]}))


@app.route("/api/live-panel", methods=["POST"])
def live_panel():
    """AI 盯盘板: 实时分时 + 模型实时信号 + 动量外推(非预测)"""
    try:
        mods = ensemble_models() or ([os.path.basename(prod_model())]
                                     if prod_model() else [])
        if not mods:
            return jsonify({"ok": False, "msg": "没有可用的最佳模型"}), 409
        body = request.get_json(silent=True) or {}
        tics = str(body.get("tics") or "").strip()
        if not tics:
            return jsonify({"ok": False, "msg": "请先输入股票代码"}), 400
        r = subprocess.run([PY, "live_panel.py", "--json", "--tics", tics,
                            "--models", ",".join(mods),
                            "--hidden", "512", "512"],
                           cwd=BASE, env=base_env(), capture_output=True,
                           text=True, timeout=120)
        lines = [x for x in r.stdout.strip().splitlines() if x.strip().startswith("{")]
        if not lines:
            return jsonify({"ok": False, "msg": "生成失败: " + r.stderr[-200:]}), 500
        data = json.loads(lines[-1])
        names = load_names()
        for t in data.get("tics", []):
            t["name"] = names.get(t["tic"], "")
        return jsonify(_safe(data))
    except Exception as e:
        return jsonify({"ok": False, "msg": "盯盘面板失败: %s" % e}), 500


@app.route("/api/intraday-demo", methods=["POST"])
def intraday_demo():
    try:
        model, hidden = prod_model(), [512, 512]
        if model is None:
            return jsonify({"ok": False, "msg": "没有可用的最佳模型"}), 409
        r = subprocess.run([PY, "intraday_decision.py", "--json",
                            "--csv", "ashare_idx2_2022_2026.csv",
                            "--model", model,
                            "--hidden"] + [str(h) for h in hidden],
                           cwd=BASE, env=base_env(), capture_output=True,
                           text=True, timeout=300)
        lines = [x for x in r.stdout.strip().splitlines() if x.strip().startswith("{")]
        if not lines:
            return jsonify({"ok": False, "msg": "生成失败: " + r.stderr[-300:]}), 500
        data = json.loads(lines[-1])
        with_names(data.get("buys"))
        with_names(data.get("sells"))
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "msg": "生成失败: %s" % e}), 500


@app.route("/api/paper/status")
def paper_status():
    acc_path = os.path.join(BASE, "paper_account_ashare.json")
    if not os.path.exists(acc_path):
        return jsonify({"started": False})
    with open(acc_path, encoding="utf-8") as f:
        acc = json.load(f)
    hist = acc.get("history", [])
    img_path = os.path.join(STATIC, "results", "paper", "paper_curve.png")
    img = "/static/results/paper/paper_curve.png" if os.path.exists(img_path) else None
    return jsonify({"started": True, "model": acc.get("model"),
                    "last_date": acc.get("last_date"),
                    "last": hist[-1] if hist else None,
                    "days": len(hist),
                    "img": img})


@app.route("/api/paper/run", methods=["POST"])
def paper_run():
    model, hidden = prod_model(), [512, 512]
    if model is None:
        for f in sorted(glob.glob(os.path.join(BASE, "trained_models", "*.pth")),
                        key=os.path.getmtime, reverse=True):
            stem = os.path.basename(f)[:-4]
            mp = os.path.join(BASE, "trained_models", stem + ".json")
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as mf:
                    meta = json.load(mf)
                if (meta.get("data_csv") or "") == "ashare_2022_2026.csv":
                    model = os.path.join("trained_models", os.path.basename(f))
                    hidden = [int(meta.get("hid", 512))] * int(meta.get("layers", 2))
                    break
    if model is None:
        return jsonify({"ok": False, "msg": "还没有 A 股模型，请先训练"}), 409
    try:
        r = subprocess.run([PY, "paper_trade.py", "--json",
                            "--csv", "ashare_idx2_2022_2026.csv",
                            "--model", model,
                            "--hidden"] + [str(h) for h in hidden],
                           cwd=BASE, env=base_env(), capture_output=True,
                           text=True, timeout=300)
        lines = [x for x in r.stdout.strip().splitlines() if x.strip().startswith("{")]
        if not lines:
            return jsonify({"ok": False, "msg": "执行失败: " + r.stderr[-300:]}), 500
        data = json.loads(lines[-1])
        with_names(data.get("orders"))
        with_names(data.get("top_positions"))
        return jsonify(_safe(data))
    except Exception as e:
        return jsonify({"ok": False, "msg": "执行失败: %s" % e}), 500


@app.route("/api/paper/reset", methods=["POST"])
def paper_reset():
    subprocess.run([PY, "paper_trade.py", "--reset"], cwd=BASE, env=base_env(),
                   capture_output=True, text=True, timeout=60)
    return jsonify({"ok": True, "msg": "模拟盘已重置"})


if __name__ == "__main__":
    print("Web 操作台已启动: http://localhost:8000")
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("waitress").setLevel(logging.INFO)
        from waitress import serve
        serve(app, host="0.0.0.0", port=8000, threads=8)
    except ImportError:
        app.run(host="0.0.0.0", port=8000, threaded=True)

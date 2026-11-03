# -*- coding: utf-8 -*-
"""
select_best.py —— 验证集选模
对某次训练的所有检查点，在验证段（默认 2025-01-01 ~ 2025-06-30）做采样评估，
选出收益最高的检查点，复制为 <name>_best.pth 并写选择报告。
用法: ~/finrl_env/bin/python select_best.py --name ashare_30ep_v2 --paths 8
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

BASE = os.path.expanduser("~/FinRL_DeepSeek")
PY = os.path.expanduser("~/finrl_env/bin/python")
CSV = "ashare_2022_2026.csv"
VAL_START, VAL_END = "2025-01-01", "2025-06-30"


def eval_ckpt(ckpt_path, hidden, paths, csv_name):
    stem = os.path.basename(ckpt_path)[:-4]
    outdir = os.path.join(BASE, "results", "ckpt_eval_" + stem)
    cmd = [PY, "use_model.py", "--model", ckpt_path, "--data-csv", csv_name,
           "--data-start", VAL_START, "--data-end", VAL_END,
           "--eval-mode", "sample", "--eval-n", str(paths), "--outdir", outdir]
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=1800)
    res = os.path.join(outdir, "回测_结果.json")
    if not os.path.exists(res):
        print("  %s: 评估失败" % stem)
        return None
    with open(res, encoding="utf-8") as f:
        s = json.load(f)
    print("  %-40s 验证收益 %8.2f%% (交易 %d)" %
          (stem, s["return_pct"], s["total_trades"]))
    return s["return_pct"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="模型前缀")
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--csv", type=str, default=CSV, help="数据 CSV")
    args = parser.parse_args()

    ckpts = sorted(glob.glob(os.path.join(BASE, "trained_models", args.name + "*.pth")))
    if not ckpts:
        print("没有找到检查点:", args.name)
        return
    print("评估 %d 个检查点 (验证段 %s ~ %s, 每点 %d 条路径, CSV %s)..."
          % (len(ckpts), VAL_START, VAL_END, args.paths, args.csv))
    scores = {}
    for ck in ckpts:
        stem = os.path.basename(ck)[:-4]
        v = eval_ckpt(ck, args.hidden, args.paths, args.csv)
        if v is not None:
            scores[ck] = v
    if not scores:
        print("全部评估失败")
        return
    best = max(scores, key=scores.get)
    best_path = os.path.join(BASE, "trained_models", args.name + "_best.pth")
    shutil.copy(best, best_path)
    report = {"best": os.path.basename(best), "best_return_pct": scores[best],
              "all": {os.path.basename(k): v for k, v in sorted(scores.items())},
              "val_range": [VAL_START, VAL_END], "paths": args.paths}
    with open(os.path.join(BASE, "trained_models", args.name + "_selection.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n最佳检查点: %s (验证收益 %.2f%%) -> 已复制为 %s_best.pth"
          % (os.path.basename(best), scores[best], args.name))


if __name__ == "__main__":
    main()

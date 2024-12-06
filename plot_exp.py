RESULT_DIR = "./results"
# MODE = "BEST_PERF"
# MODE = "AGG_DATA"
MODE = "PLOT_DSE"


BENCHMARK_SUITES = [
#   "gemm-p",
  "bradybench_0",
  "bradybench_1",
  "bradybench_2",
  "bradybench_5",
  "bradybench_7",
  "bradybench_9",
  "bradybench_16",
  "bradybench_17",
  "bradybench_18",
  "bradybench_19",
#   "cnn"
]

HARP_BASELINE = {
    "bradybench_0": 39347,
    "bradybench_1": 90677,
    "bradybench_2": 2960,
    "bradybench_5": 260479,
    "bradybench_7": 26339203,
    "bradybench_9": 1270907,
    "bradybench_16": 1047849,
    "bradybench_17": 5585,
    "bradybench_18": 6541,
    "bradybench_19": 2720791
}

INT_MAX = 2**31 - 1
import pandas as pd
import numpy as np
import re

def extract_parathesis(s):
    return int(re.search(r'\((.*?)\)', s).group(1).replace("~", "").replace("%", ""))/100 if isinstance(s, str) and "(" in s else INT_MAX
def exclude_parathesis(s):
    return int(s.split("(")[0].strip()) if isinstance(s, str) and "(" in s else INT_MAX

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
BENCHMARKS_TO_PLOT = [
    "bradybench_2",
    "bradybench_18",
    "bradybench_19"
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
curr_ax = 0

df = pd.DataFrame()
for bmark in BENCHMARK_SUITES:
    _df = pd.read_csv(f"{RESULT_DIR}/{bmark}.csv")
    _df["benchmark"] = bmark
    _df["cycles"] = _df["cycles"].apply(exclude_parathesis)
    for k in ["lut utilization", "DSP utilization", "FF utilization", "BRAM utilization", "URAM utilization"]:
        _df[k] = _df[k].apply(extract_parathesis)
    _df["max util."] = _df[["lut utilization", "DSP utilization", "FF utilization", "BRAM utilization", "URAM utilization"]].max(axis=1)
    _df["perf"] = _df["cycles"]
    _df.loc[_df["max util."] > 0.8, "perf"] = INT_MAX
    if MODE == "BEST_PERF":
        best_perf = _df["perf"].min()
        print(f"{bmark},{best_perf}")
    elif MODE == "PLOT_DSE":
        if bmark not in BENCHMARKS_TO_PLOT: continue
        _df["Util. Ratio"] = _df["max util."]
        _df["Norm. Perf"] = np.log(_df["cycles"]/HARP_BASELINE[bmark]) + 1
        _df["Index"] =  _df["step"]
        _df = _df[_df["cycles"] != INT_MAX].sort_values("step")
        axes[curr_ax].scatter(_df["Util. Ratio"], _df["Norm. Perf"], c="blue", s=50, marker="s")
        # plot the arrow to indicate the trajectory based on the step
        for i in range(1, len(_df)):
            axes[curr_ax].annotate("", xy=(_df["Util. Ratio"].iloc[i], _df["Norm. Perf"].iloc[i]), xytext=(_df["Util. Ratio"].iloc[i-1], _df["Norm. Perf"].iloc[i-1]), arrowprops=dict(arrowstyle="->, head_width=0.25, head_length=0.5", color="black", lw=1, ls="-", alpha=0.9, ))
        # plot step labels next to the data points
        # for i in range(len(_df)):
        #     plt.text(_df["Util. Ratio"].iloc[i], _df["Norm. Perf"].iloc[i], f"{_df['step'].iloc[i]}", fontsize=10)
        axes[curr_ax].set_ylim(0, max(_df["Norm. Perf"])+1)
        axes[curr_ax].axhline(y=1, color="blue", linestyle="--", lw=1)
        axes[curr_ax].text(0.2, 1 + max(_df["Norm. Perf"])/100, "HARP's DSE Perf.", fontsize=10, rotation=0, color="blue")
        axes[curr_ax].set_xlim(0, 1)
        # plot a red hashline to indicate the 80% utilization
        axes[curr_ax].axvline(x=0.8, color="red", linestyle="--", lw=1)
        axes[curr_ax].text(0.8, max(_df["Norm. Perf"])/2, "80% Max. Util.", fontsize=10, rotation=270, color="red")
        
        axes[curr_ax].set_xlabel("Utilization Ratio (Max. of LUT, DSP, FF, BRAM, URAM)")
        axes[curr_ax].set_ylabel("Normalized Performance: log(#Cycles) + 1")
        axes[curr_ax].set_title(f"{bmark}")
        curr_ax += 1
    elif MODE == "AGG_DATA":
        df = pd.concat([df, _df])

if MODE == "PLOT_DSE":
    plt.savefig(f"{RESULT_DIR}/all.pdf")       

if MODE == "AGG_DATA":
    df.to_csv(f"{RESULT_DIR}/all.csv", index=False)
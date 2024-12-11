# retrieve the designs from the bradybench experiments
RESULT_DIR = "/scratch/hanyu/work_bradybench_19_20241210_143204"

import os
import pandas as pd
from util import *
datas = []
for i in range(10):
    merlin_rpt = os.path.join(RESULT_DIR, f"{i}", "merlin.rpt")
    if not os.path.exists(merlin_rpt):
        print(f"Missing {merlin_rpt}")
        perf = {}
    else: perf = parse_merlin_rpt(merlin_rpt)
    datas.append(perf if perf is not None else {})
pd.DataFrame(datas).to_csv("bradybench.csv", index=False)
LLM-DSE
=======

Getting Started
---------------

**Step 1** Preparation:
Ensure that you have 300 GB of free space on your device.
- Docker: https://docs.docker.com/engine/install/
- AMD/Xilinx Vitis HLS: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html

To reproduce the code, you need to pull and run the docker from VAST DSE-for-HLS.
`https://github.com/UCLA-VAST/DSE-for-HLS/tree/main/example`

```bash
./merlin.sh
Apptainer> source run.sh
```


**Step 2** Before running the script, you need to put the OPENAI API key in the `.env` file and specify a work dir (all itermediate results will be stored there).
```bash
cp .env_example .env
```

**Step 3** Then, you can run the script with the following command:
```bash
pip install -r requirements.txt
python main.py --benchmark 3mm --folder ./data/benchmarks
```

Troubleshooting
---------------

> Can't install Vitis and Merlin Compiler?

Toggle the `ENABLE_DATABASE_LOOKUP` flag in `config.py` to `True` and run the script again. This will look up the compilation history in the database and is likely to skip the compilation step.

> Can't find it in the database?

Toggle the `DEBUG_MERLIN` flag in `config.py` to `True` and run the script again. This will skip the compilation step and use a fake backend. 

> No OPENAI API key or no budget?

Toggle the `AUTO_<XX>` flag in `config.py` (replace `XX` with the agent names) to bypass the agent using heuristics. This will skip the agent and use a fake backend.

import os

for filename in os.listdir('.'):
    if filename.endswith('_kernel.c'):
        # rename it to X.c
        os.rename(filename, filename.replace('_kernel.c', '.c'))
    elif filename.endswith('_ds_config.json'):
        # rename it to X.json
        os.rename(filename, filename.replace('_ds_config.json', '.json'))
import os
import re

folder_name = './data/lad25'

def label_c(file_in, file_out):
    lines = open(file_in, 'r').readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#pragma ACCEL') and any([x in line for x in ['PIPE', 'TILE', 'PARA']]):
            label = re.search(r'auto\{(\w+)\}', line).group(1).split('__')[-1]
            while i < len(lines) and not lines[i].strip().startswith('for'): i += 1
            if i < len(lines): lines[i] = f'{label}: {lines[i]}'
        i += 1
    open(file_out, 'w').writelines(lines)
            
if __name__ == "__main__":
    for filename in os.listdir(folder_name):
        if filename.endswith(".c"):
            label_c(os.path.join(folder_name, filename), os.path.join(folder_name, filename))
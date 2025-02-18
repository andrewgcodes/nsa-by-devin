import os

def fix_whitespace(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = [line.rstrip() + '\n' for line in lines]
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

for root, _, files in os.walk('nsa'):
    for file in files:
        if file.endswith('.py'):
            fix_whitespace(os.path.join(root, file))

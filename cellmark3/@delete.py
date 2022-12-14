import os
import glob

basename_excludes = [
    "__pycache__",
    "@delete.py"
]

source_files=[]

for file in glob.glob("**", recursive=True):
    print(file)
    if (file.endswith(".js") or file.endswith(".py")) and (os.path.basename(file) not in basename_excludes):
        filepath = os.path.normpath(os.path.join(".",file))
        source_files.append(filepath)
        
print(f"Found Files:\n"+'\n'.join(source_files) if source_files else 'No Files')

for source_file in source_files:
    print(f"Cleaning file \"{source_file}\"...")
    content = ""
    with open(source_file, "r") as fl:
        content = fl.read()
    lines = content.split("\n")
    filtered_lines = []
    for idx, line in enumerate(lines):
        if (not "//@delete" in line) and (not "#@delete" in line):
            filtered_lines.append(line)
        else:
            print(f"Deleted line {idx+1} in file {source_file}: ```{line}```")
    content = "\n".join(filtered_lines)
    with open(source_file, "w") as fl:
        fl.write(content)


                
    
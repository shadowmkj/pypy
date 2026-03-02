from pathlib import Path

source_folder = Path("./data")
files = [p for p in source_folder.iterdir()]

for f in files:
    if f.is_file() and f.match("*.pdf"):
        print(f.name)

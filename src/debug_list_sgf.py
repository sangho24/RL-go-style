from pathlib import Path

root = Path("data/raw/cwi/games")

print("[DEBUG] root:", root, "exists?", root.exists())

sgf_files = list(root.rglob("*.sgf")) + list(root.rglob("*.SGF"))

print("[DEBUG] number of sgf files found:", len(sgf_files))
for f in sgf_files[:10]:
    print("   ", f)

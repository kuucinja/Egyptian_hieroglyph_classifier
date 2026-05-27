from pathlib import Path
import ast

ROOT = Path(".")

def get_functions(file_path):
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    except:
        return []

output = []

for path in ROOT.rglob("*"):
    if path.is_dir():
        continue

    output.append(f"\nFILE: {path}")

    if path.suffix == ".py":
        funcs = get_functions(path)
        if funcs:
            output.append("  FUNCTIONS:")
            for f in funcs:
                output.append(f"    - {f}")

with open("project_structure.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved to project_structure.txt")
import os
import ast

def summarize_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading {filepath}: {e}"
        
    try:
        tree = ast.parse(content)
    except Exception as e:
        return f"Syntax error in {filepath}: {e}"
        
    summary = []
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    
    if not classes and not funcs:
        return ""
        
    summary.append(f"--- {filepath} ---")
    for c in classes:
        methods = [m.name for m in c.body if isinstance(m, ast.FunctionDef)]
        summary.append(f"Class: {c.name}")
        if methods:
            summary.append(f"  Methods: {', '.join(methods)}")
            
    for f in funcs:
        summary.append(f"Function: {f.name}")
        
    return "\n".join(summary)

base_dir = "."
results = []
for root, _, files in os.walk(base_dir):
    for f in sorted(files):
        if f.endswith(".py") and f != "__init__.py":
            res = summarize_file(os.path.join(root, f))
            if res:
                results.append(res)
                
with open("summary.txt", "w") as f:
    f.write("\n\n".join(results))
print("Done")

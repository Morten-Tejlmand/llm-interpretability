import subprocess
import sys

PY = sys.executable

def run_script(name):
    print(f"\n=== Running {name} ===")
    result = subprocess.run(
        [PY, name],
        text=True,
        capture_output=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"\n[{name}] crashed with exit code {result.returncode}")
        print("Error output:")
        print(result.stderr)
        raise SystemExit(1)

run_script("main.py")
run_script("preds.py")

import os, subprocess
print("ENV", {k: v for k, v in os.environ.items() if "KAGGLE" in k})
for root in ["/kaggle/input", "/kaggle/input/competitions", "/kaggle/input/datasets"]:
    print("LS", root, sorted(os.listdir(root)) if os.path.isdir(root) else "MISSING")
print(subprocess.run("find /kaggle/input -maxdepth 4 -name 'arc_agi-*.whl' -o -maxdepth 4 -type d -name arc_agi_3_wheels", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run("mount | grep -i kaggle | head -20", shell=True, capture_output=True, text=True).stdout)

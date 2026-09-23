# Grade the critic-v2 questioner on frozen packets: start vLLM (Duck's own setup script), ask, save.
import glob, json, os, shutil, subprocess, sys
W = "/kaggle/working"
src = os.path.dirname(glob.glob("/kaggle/input/**/grade.py", recursive=True)[0])
os.makedirs(W + "/grade", exist_ok=True)
for f in ("grade.py", "questioner.py", "packets.json", "setup.sh"):
    shutil.copy(os.path.join(src, f), W + "/grade/" + f)
env = {**os.environ, "PYTHON": sys.executable, "TAAF_KAGGLE_SETUP_ENV": W + "/setup_env.json", "TAAF_KAGGLE_WORKING_DIR": W}
rc = subprocess.run(["bash", W + "/grade/setup.sh"], env=env).returncode
print("setup rc", rc, flush=True)
if rc == 0:
    subprocess.run([sys.executable, W + "/grade/grade.py", "Qwen/Qwen3.8-27B-FP8"], env={**os.environ, "GRADE_BASE": W})

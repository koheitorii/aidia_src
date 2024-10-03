import subprocess
from aidia import __version__, IS_LITE

if IS_LITE:
    fname = f"dist/aidia-{__version__}-py3-none-any.whl"
    subprocess.run(["pip", "uninstall", "aidia-lite", "-y"], encoding="utf-8")
else:
    fname = f"dist/aidia-lite-{__version__}-py3-none-any.whl"
    subprocess.run(["pip", "uninstall", "aidia", "-y"], encoding="utf-8")

subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], encoding="utf-8")
subprocess.run(["pip", "install", fname], encoding="utf-8")
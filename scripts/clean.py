import shutil
from pathlib import Path
from omegaconf import OmegaConf

RUN_DIR = Path("../runs/")
sweep_id = 27

for run in RUN_DIR.glob("*"):
    if not run.is_dir():
        continue
    if (run / "config.yaml").exists():
        config = OmegaConf.load(run / "config.yaml")
        if config.get("sweep_id", None) == sweep_id:
            print(f"Removing {run}...")
            shutil.rmtree(run)
from __future__ import annotations
import argparse,json
from pathlib import Path
from .common import TRAINING_MODES
from .partitioned import run_cli
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=TRAINING_MODES,default="Preflight");p.add_argument("--project-root",type=Path,default=Path.cwd());p.add_argument("--config",type=Path);p.add_argument("--run-name",default="");a=p.parse_args();print(json.dumps(run_cli("receraser",a),indent=2,sort_keys=True))
if __name__=="__main__":main()

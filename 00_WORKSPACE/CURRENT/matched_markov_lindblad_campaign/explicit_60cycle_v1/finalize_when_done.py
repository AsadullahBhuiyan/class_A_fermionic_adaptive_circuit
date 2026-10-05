"""One-shot tmux companion: analyze only after both workers finish successfully."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_dynamics import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    deadline=time.monotonic()+6*3600
    print('[waiting] two worker completion records; six-hour limit',flush=True)
    while time.monotonic()<deadline:
        paths=[root/f'worker_alpha{a}.json' for a in (1,3)]
        if all(path.exists() for path in paths):
            states=[json.loads(path.read_text()) for path in paths]
            if any(s['complete']!=8 or s['failures'] for s in states):
                atomic_json(root/'finalization.json',dict(status='incomplete',workers=states))
                return 1
            command=[sys.executable,str(Path(__file__).with_name('analyze_dynamics.py')),'--root',str(root)]
            result=subprocess.run(command)
            atomic_json(root/'finalization.json',dict(status='complete' if result.returncode==0 else 'analysis_failed',
                analysis_returncode=result.returncode,workers=states))
            return result.returncode
        time.sleep(10)
    atomic_json(root/'finalization.json',dict(status='wait_timeout'))
    return 1


if __name__=='__main__':
    raise SystemExit(main())

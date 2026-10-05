"""Launch exactly four endpoint processes in detached tmux sessions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--cpus',default='8,9,10,11')
    parser.add_argument('--hard-exterior',choices=('evolve','frozen'),default='evolve')
    args=parser.parse_args()
    cpus=[int(x) for x in args.cpus.split(',')]
    if len(cpus)!=4 or len(set(cpus))!=4:
        raise ValueError('Specify four distinct physical CPUs after checking utilization')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    revision='alpha3_full_system_dissipation_v1' if args.hard_exterior=='evolve' else 'alpha3_frozen_exterior_v1'
    root=HERE/'results'/f'{revision}_{stamp}'
    jobs=[]
    for cpu,(family,wall) in zip(cpus,[(f,w) for f in ('markov_channel','lindblad') for w in ('soft','hard')]):
        label=family+'_'+wall
        command=['nice','-n','10',sys.executable,'-u',str(HERE/'run_alpha3_endpoint.py'),
            '--family',family,'--wall',wall,'--hard-exterior',args.hard_exterior,
            '--cpu',str(cpu),'--run-root',str(root/label)]
        session=f'a3_{family}_{wall}_{stamp}'
        log=root/(label+'.log')
        jobs.append(dict(family=family,wall=wall,cpu=cpu,session=session,
                         command=command,log=str(log),output=str(root/label)))
    if args.launch:
        root.mkdir(parents=True,exist_ok=False)
        for job in jobs:
            shell='set -o pipefail; '+shlex.join(job['command'])+' 2>&1 | tee '+shlex.quote(job['log'])
            subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(HERE),
                            'bash','-lc',shell],check=True)
        (root/'launch.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps({'launched':args.launch,'root':str(root),'jobs':jobs},indent=2))


if __name__=='__main__':
    main()

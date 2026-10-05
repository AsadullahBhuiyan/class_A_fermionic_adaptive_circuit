"""Launch hard/soft alpha_1=1/3 raster-y channel endpoints, preserving old runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cpus', default='8,9,10,11')
    args = parser.parse_args()
    cpus = [int(x) for x in args.cpus.split(',')]
    if len(cpus) != 4 or len(set(cpus)) != 4:
        raise ValueError('Specify four distinct CPUs after checking utilization')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = HERE / 'results' / f'raster_y_channel_endpoints_v1_{"smoke_" if args.smoke else ""}{stamp}'
    jobs = []
    for cpu, (alpha, wall) in zip(cpus, [(a, w) for a in (1, 3) for w in ('hard', 'soft')]):
        label = f'alpha{alpha}_{wall}'
        command = ['nice', '-n', '10', sys.executable, '-u', str(HERE / 'run_alpha3_endpoint.py'),
                   '--family', 'markov_channel', '--wall', wall, '--alpha-1', str(alpha),
                   '--sequence', 'raster_y', '--hard-exterior', 'evolve', '--cpu', str(cpu),
                   '--run-root', str(root / label)]
        if args.smoke:
            command.append('--smoke')
        jobs.append(dict(alpha_1=alpha, wall=wall, cpu=cpu, command=command,
                         session=f'rastery_{label}_{stamp}', log=str(root / f'{label}.log'),
                         output=str(root / label)))
    if args.launch:
        root.mkdir(parents=True, exist_ok=False)
        (root / 'launch.json').write_text(json.dumps(jobs, indent=2) + '\n')
        for job in jobs:
            shell = 'set -o pipefail; ' + shlex.join(job['command']) + ' 2>&1 | tee ' + shlex.quote(job['log'])
            subprocess.run(['tmux', 'new-session', '-d', '-s', job['session'], '-c', str(HERE),
                            'bash', '-lc', shell], check=True)
    print(json.dumps(dict(launched=args.launch, root=str(root), jobs=jobs), indent=2))


if __name__ == '__main__':
    main()

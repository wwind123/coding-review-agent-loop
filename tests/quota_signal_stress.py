import argparse, json, os, signal, subprocess, sys, time, importlib.metadata
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--iterations',type=int,default=50); p.add_argument('--workers',type=int,required=True); p.add_argument('--per-run-timeout',type=int,default=120); p.add_argument('--label',required=True); a=p.parse_args()
root=Path('/tmp/coding-review-agent-loop/scratch')/f'{a.label}-{a.workers}'; root.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('PYTEST_', 'CI_SHARD_')) and k not in {'SHARDS','SHARD_INDEX','DURATIONS_FILE','TEST_COMMAND'}}
command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','-p','no:randomly','-n',str(a.workers),'tests/test_quota_reset_wait.py']
summary={'command':command,'python':sys.version,'plugins':{x:importlib.metadata.version(x) for x in ['pytest','pytest-xdist','pytest-timeout']},'attempts':0,'passes':0,'failures':0,'timeouts':0}; start=time.monotonic()
for i in range(a.iterations):
    with (root/f'{i+1:03}.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=proc.wait(timeout=a.per_run_timeout)
        except subprocess.TimeoutExpired:
            summary['timeouts']+=1; os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=10); code=-9
        finally:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
    summary['attempts']+=1; summary['passes' if code==0 else 'failures']+=1
    print(f'{a.label} workers={a.workers} iteration={i+1} exit={code}',flush=True)
summary['elapsed_seconds']=time.monotonic()-start
(root/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
sys.exit(bool(summary['failures']))

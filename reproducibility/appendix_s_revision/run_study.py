#!/usr/bin/env python3
"""Replay the frozen suite/corpus; no model calls. Use a fresh output directory."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--python',default=sys.executable);p.add_argument('--gpus',default='0');p.add_argument('--out',required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
for name,digest in json.loads((root/'manifest.json').read_text()).items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Snapshot hash mismatch: '+name)
out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
ops=json.loads((root/'repeat_selection.json').read_text())['operators']
# All 152 candidates in baseline and held-out. Repeats 2/3 cover the
# predeclared 21-task subset; baseline is repeat 1 for those tasks.
clone_ops=[o for o in ops if o.startswith('aten::') and not o.endswith('_')]
arms=[('baseline','baseline',None),('heldout','heldout',None),('repeat2','baseline',ops),('repeat3','baseline',ops),('clone_free','clone_free',clone_ops)]
base=[args.python,str(root/'snapshot/scripts/analyze/reverify_corpus.py'),'--kernels',str(root/'corpus'),'--gpus',args.gpus,'--jobs',str(len(args.gpus.split(','))),'--python',args.python,'--timeout','900']
# A zero-output bmm must fail numerically, not merely fail to compile/import.
control=base.copy()
control[control.index('--kernels')+1]=str(root/'negative_control')
control+=['--policy','baseline','--out',str(out/'negative_control.json'),'--verify-dir',str(out/'logs/negative_control')]
print('START negative_control',flush=True)
with (out/'negative_control.log').open('w') as log:
 subprocess.run(control,stdout=log,stderr=subprocess.STDOUT,check=True)
rows=json.loads((out/'negative_control.json').read_text())['results']
assert len(rows)==1 and not rows[0]['passed'] and rows[0]['total_tests']==9
assert 'Tensor-likes are not close' in (rows[0].get('error') or ''), 'Negative control did not fail numerically'
print('DONE negative_control: correctly rejected',flush=True)
for name,policy,subset in arms:
 cmd=base+['--policy',policy,'--out',str(out/(name+'.json')),'--verify-dir',str(out/'logs'/name)]
 if name=='baseline':cmd+=['--audit-dir',str(out/'audit')]
 if subset:cmd+=['--only',','.join(subset)]
 print('START '+name,flush=True)
 with (out/(name+'.log')).open('w') as log:
  subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 print('DONE '+name,flush=True)

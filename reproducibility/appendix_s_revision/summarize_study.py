#!/usr/bin/env python3
"""Summarize frozen replay outputs without GPU or model calls."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();root=Path(__file__).resolve().parent;out=Path(a.results)
def load(name):return {r['op']:r for r in json.loads((out/(name+'.json')).read_text())['results']}
b=load('baseline');h=load('heldout');r2=load('repeat2');r3=load('repeat3');c=load('clone_free');selected=json.loads((root/'repeat_selection.json').read_text())['operators']
assert set(b)==set(h) and len(b)==152
assert all(set(x)==set(selected) for x in [r2,r3])
assert set(c).issubset(selected)
accepted={o for o in b if b[o].get('passed')}; held_retained={o for o in accepted if h[o].get('passed')}
added={o for o in b if h[o].get('total_tests',0)>b[o].get('total_tests',0)}
audits={o:b[o]['tolerance'] for o in accepted if b[o].get('tolerance')}
timing=[];changed=[];signatures=[]
for o in selected:
 rows=[x[o] for x in [b,r2,r3]];statuses=[bool(r.get('passed')) for r in rows]
 if len(set(statuses))>1:changed.append(o)
 if len({(r.get('passed_tests'),r.get('total_tests')) for r in rows})>1:signatures.append(o)
 values=[r.get('speedup') for r in rows]
 if all(statuses) and all(isinstance(v,(float,int)) and v>0 for v in values):
  timing.append({'operator':o,'speedups':values,'relative_range':max(values)/min(values)-1,'cv':statistics.stdev(values)/statistics.mean(values)})
pairs=[{'operator':o,'baseline':b[o]['speedup'],'clone_free':c[o]['speedup']} for o in c if b[o].get('passed') and c[o].get('passed') and all(isinstance(x.get('speedup'),(float,int)) and x['speedup']>0 for x in [b[o],c[o]])]
gm=lambda xs: math.exp(statistics.mean(math.log(x) for x in xs)) if xs else None
summary={'scope':'Frozen candidate functional verification; no new generation and no runtime L2/L3 anti-hack audit.','total':len(b),'baseline_passed':len(accepted),'heldout_passed':sum(bool(x.get('passed')) for x in h.values()),'accepted_retained_heldout':len(held_retained),'new_test_cases_all':len(added),'new_test_cases_accepted':len(added&accepted),'tolerance_audited_accepted':len(audits),'tolerance_counts':{k:sum(bool(t[k]) for t in audits.values()) for k in ['passes_scaled','passes_sqrt','passes_const']},'tolerance_comparisons':sum(t['n_checks'] for t in audits.values()),'tolerance_max_reduce_dim':max((t['max_reduce_dim'] for t in audits.values()),default=0),'repeat_tasks':len(selected),'repeat_runs':3,'repeat_pass_counts':[sum(bool(x[o].get('passed')) for o in selected) for x in [b,r2,r3]],'changed_outcomes':changed,'changed_case_counts':signatures,'repeat_timings':timing,'clone_pairs':pairs,'clone_ratio_geomean':gm([x['clone_free']/x['baseline'] for x in pairs]),'baseline_failures':[{k:b[o].get(k) for k in ['op','passed_tests','total_tests','error']} for o in sorted(set(b)-accepted)]}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Frozen-snapshot replay results','',summary['scope'],'',f"- Baseline: {len(accepted)}/{len(b)} pass.",f"- Held-out: {len(held_retained)}/{len(accepted)} baseline successes retained; {len(added&accepted)} accepted tasks receive additional test cases.",f"- Three repeats on {len(selected)} predeclared tasks: {summary['repeat_pass_counts']}; {len(changed)} tasks change pass/fail.",f"- {len(signatures)} tasks change passed/total case counts.",f"- Accepted tasks with floating comparison audits: {len(audits)}; rule counts: {summary['tolerance_counts']}.",'','## Repeated timings on tasks passing all three runs','','Timing is noisy; ranges below are observed variation, not confidence intervals or a guarantee of identical future runs.','','| Operator | Run 1 | Run 2 | Run 3 | Relative range |','|---|---:|---:|---:|---:|']
for x in timing:lines.append('| '+x['operator']+' | '+' | '.join(f'{v:.4f}' for v in x['speedups'])+f" | {x['relative_range']:.1%} |")
lines+=['','## Changed functional outcomes','',', '.join(changed) or 'None in this selected subset.','', '## Clone-free comparison','',f"Successful in both arms with measured speedups: {len(pairs)}; geometric mean ratio: {summary['clone_ratio_geomean']}",'','## Interpretation','','This is a fixed-code replay study. It does not estimate LLM/agent generation variance, prove full API correctness, or reproduce historical main-table clean rates. The subset is deliberately selected, not a random sample. Frozen code, inputs and environment permit re-execution; measured timing need not match exactly.']
(out/'summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['baseline_failures','repeat_timings','clone_pairs']},indent=2))

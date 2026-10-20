"""Execute the supplied assertion functions when pytest is unavailable.
Only the supplied tests' parameterization and raises features are emulated.
This is not a pytest run; no package code or tests are changed.
"""
import importlib.util, itertools, json, sys, tempfile, time, traceback, types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'provenance'
sys.path.insert(0,str(ROOT/'src'))
class Raises:
 def __init__(self,typ): self.typ=typ
 def __enter__(self):return self
 def __exit__(self,t,v,tb):
  if t is None:raise AssertionError('Expected exception was not raised')
  return issubclass(t,self.typ)
def parametrize(names,values):
 def decorate(fn):
  fn._audit_params=(names.split(','),values);return fn
 return decorate
sys.modules['pytest']=types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=parametrize),raises=Raises)
rows=[]
for file in sorted((ROOT/'tests').glob('test_*.py')):
 spec=importlib.util.spec_from_file_location(file.stem,file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for name,fn in sorted(vars(mod).items()):
  if not name.startswith('test_') or not callable(fn):continue
  params=getattr(fn,'_audit_params',None)
  if params:
   names,vals=params;calls=[dict(zip(names,(v if len(names)>1 else (v,)))) for v in vals]
  else:calls=[{}]
  for kwargs in calls:
   label=f'{file.name}:{name}({kwargs})';start=time.time()
   try:
    if fn.__code__.co_varnames[:fn.__code__.co_argcount]==('tmp_path',):
     with tempfile.TemporaryDirectory(dir=OUT) as t:fn(Path(t))
    else:fn(**kwargs)
    result='passed';error=None
   except Exception as e:result='failed';error=traceback.format_exc()
   rows.append(dict(test=label,result=result,seconds=time.time()-start,error=error));print(result,label,flush=True)
(OUT/'final_assertion_results.json').write_text(json.dumps(dict(runner='direct execution of parameter-expanded assertion cases; not a pytest invocation',passed=sum(r['result']=='passed' for r in rows),failed=sum(r['result']=='failed' for r in rows),tests=rows),indent=2))
print('SUMMARY',len(rows),sum(r['result']=='failed' for r in rows),'failed',flush=True)

if any(row["result"]=="failed" for row in rows):
 raise SystemExit(1)

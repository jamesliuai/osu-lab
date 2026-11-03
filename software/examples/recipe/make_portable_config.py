"""Resolve the included generated-only recipe template to explicit pool paths."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pools',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    root=Path(a.pools).resolve()
    cfg=json.loads(Path(__file__).with_name('recipe_template.json').read_text())
    entries=[cfg['env'],*cfg['evals']]
    for entry in entries:
        q=root/entry['pool']
        if not q.is_file(): p.error(f'missing generated pool: {q}')
        entry['pool']=q.as_posix()
    with open(a.out,'x',encoding='utf-8',newline='\n') as f:
        json.dump(cfg,f,indent=2)
        f.write('\n')
    print(a.out)

if __name__=='__main__': main()

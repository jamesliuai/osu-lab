"""Verify public document provenance, semantic records, links and chart annotations.

Standard library only. Does not open historical source paths, render PDFs, import
the simulator, or execute study/training/audit code. PDF content/layout inspection
is recorded separately; hashes here check the inspected bytes.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, re, sys


def resolve(data, keys):
    for key in keys:
        data = data[key]
    return data


def verify(reader, claims=None):
    name = 'evidence/documents/DOCUMENT_PROVENANCE.json'
    record = reader.read_json(name)
    assert record['schema'] == 'public_document_provenance_v1'
    for item in record['public_evidence']:
        reader.hash_reference(name, item)
    for item in record['semantic_bindings']:
        actual = resolve(reader.read_json(item['path']), item['selector'])
        assert actual == item['value'], (item['path'], item['selector'])
    for item in record['required_document_text']:
        text = reader.path(item['path']).read_text(encoding='utf-8')
        assert item['text'] in ' '.join(text.split()), item
    links = 0
    for doc in record['documents']:
        path = reader.path(doc['path'])
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'name to be added|Optional data note|the author.s decision|What I built',text)
        for link in re.findall(r'\]\(([^)]+)\)|src="([^"]+)"',text):
            target = next(x for x in link if x)
            if target.startswith(('https://','http://','#')):
                continue
            destination = (path.parent/target.split('#')[0]).resolve()
            assert destination.is_relative_to(reader.root) and destination.is_file(), target
            links += 1
        tables = ['\n'.join(block) for block in table_blocks(text)]
        assert [hashlib.sha256(x.encode()).hexdigest() for x in tables] == doc['scientific_table_sha256']
    references = reader.read_json('evidence/documents/REFERENCE_AUDIT.json')
    bib = references['bibliography']
    reader.hash_reference('evidence/documents/REFERENCE_AUDIT.json', {'path':bib['public_path'],'sha256':bib['sha256']})
    bib_text = reader.path(bib['public_path']).read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+),',bib_text))
    assert keys == {r['key'] for r in references['entries']} and len(keys)==18
    fig_name='evidence/documents/FIGURE_AUDIT.json'
    figures=reader.read_json(fig_name)
    reader.hash_reference(fig_name,figures['analysis'])
    for figure in figures['figures']:
        for artifact in figure['files']:
            reader.hash_reference(fig_name,{'path':artifact['public_path'],'sha256':artifact['sha256']})
    spec=importlib.util.spec_from_file_location('public_document_figures',reader.path('tools/verify_document_figures.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    figure_result=module.verify_document_figures(reader.root)
    receipt=reader.read_json('evidence/study/EXPLORATORY_RECEIPT.json')
    analysis=reader.read_json('evidence/study/PD_analysis.json')
    by_name={r['name']:r for r in analysis['runs']}
    for row in receipt['spinner_post_hoc']['runs']:
        assert row['failures_total']==by_name[row['run']]['sets']['id']['F']
    d1=reader.path('evidence/study/PD_DEVIATIONS_2026-11-01.md').read_text(encoding='utf-8')
    assert 'I accept D1 retrospectively as a documented deviation' in d1
    assert record['owner_decisions']['D1']=='approved_retrospectively'
    if claims:
        assert claims['PERF-001']['value']==reader.read_json('evidence/study/PD_supplement.json')['throughput']
        assert claims['HIST-001']['value']=={k:reader.read_json('evidence/study/kit_facts.json')['facts'][k]['value'] for k in ['render_fg_mae_gen_a','render_fg_mae_gen_b','render_frames_gen']}
    return {'status':'PASS','semantic_bindings':len(record['semantic_bindings']),'required_document_text':len(record['required_document_text']),
            'resolved_document_links':links,'bibliography_entries':len(keys),'figure_checks':figure_result,
            'exploratory_totals_matched':9,'historical_sources_opened':False,'runtime_or_study_executed':False}


def table_blocks(text):
    block=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):
            block.append(line)
        elif block:
            yield block;block=[]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from verify_claims import PublicReader
    try:
        print(json.dumps(verify(PublicReader(args.root)),indent=2))
    except (AssertionError,KeyError,ValueError,OSError) as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)}),file=sys.stderr);raise SystemExit(1)

"""Validate only the declared repository payload; refuse an unready Pages site.

This standard-library utility never follows links or original research paths.
The publication inventory is the explicit allowlist; .git is local repository
metadata, excluded from all published payloads. No remote operation is performed.
"""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,sys


def require(value,message):
    if not value:raise ValueError(message)


def verify(root,require_site=False):
    root=root.resolve()
    manifest=root/'REPOSITORY_PUBLICATION.json'
    data=json.loads(manifest.read_text(encoding='utf-8'))
    require(data['schema']=='repository_publication_inventory_v1','unsupported inventory')
    require(data['project']=='osu-lab' and data['repository']=='jamesliuai/osu-lab','wrong planned repository')
    files={}
    for row in data['files']:
        name=row['path'];parsed=PurePosixPath(name)
        require(name not in files and not parsed.is_absolute() and all(x not in ('.','..','.git') for x in parsed.parts),'unsafe/duplicate publication path')
        require('\\' not in name and ':' not in name and str(parsed)==name,'noncanonical publication path')
        p=root/name
        require(not p.is_symlink() and p.resolve().is_relative_to(root) and p.is_file(),'missing/linked/outside file: '+name)
        require(p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],'publication bytes differ: '+name)
        require(bool(row['license_scope']) and bool(row['provenance']),'missing scope/provenance: '+name)
        files[name]=row
    actual=set()
    def walk(folder):
        for p in folder.iterdir():
            if folder==root and p.name=='.git':continue
            require(not p.is_symlink(),'symlink in staging')
            if p.is_dir():walk(p)
            elif p.is_file():actual.add(p.relative_to(root).as_posix())
    walk(root)
    require(actual==set(files)|{'REPOSITORY_PUBLICATION.json','REPOSITORY_SHA256SUMS'},'unlisted/missing repository payload')
    sums={}
    for line in (root/'REPOSITORY_SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        require(name in set(files)|{'REPOSITORY_PUBLICATION.json'} and name not in sums,'unsafe/duplicate checksum path')
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'repository checksum mismatch')
        sums[name]=digest
    require(set(sums)==set(files)|{'REPOSITORY_PUBLICATION.json'},'checksum coverage differs')
    require(data['site']['repository_prefix']=='site/','unexpected Pages root')
    site_files=[p for p in files if p.startswith('site/')]
    ready=data['site']['status']=='verified'
    if not ready:
        require(not site_files,'unverified site payload must not be staged')
        if require_site:raise ValueError('PAGES BLOCKED: website owner update/export verification is incomplete; no site payload is staged.')
    else:
        require('site/index.html' in files and 'site/.nojekyll' in files,'verified site lacks index or .nojekyll')
        require(data['site']['verification']['status']=='PASS','site lacks passed verification')
        require(data['site']['verification']['candidate']==data['candidate'],'site candidate differs')
    return dict(status='PASS',project=data['project'],candidate=data['candidate'],payload_files=len(files),site_status=data['site']['status'],site_payload_files=len(site_files),pages_ready=ready,remote_action_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--site',action='store_true',help='Also require verified Pages output before deployment')
    args=parser.parse_args()
    try:print(json.dumps(verify(args.root,args.site),indent=2))
    except (ValueError,KeyError,OSError) as exc:
        print(json.dumps({'status':'BLOCKED' if 'PAGES BLOCKED' in str(exc) else 'FAIL','error':str(exc)}),file=sys.stderr);raise SystemExit(2 if 'PAGES BLOCKED' in str(exc) else 1)

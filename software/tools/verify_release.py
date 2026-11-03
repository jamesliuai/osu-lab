"""Verify the public source payload using its manifest and SHA256SUMS (standard library only)."""
from pathlib import Path
import argparse, hashlib, json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--strict',action='store_true',help='also reject unlisted files; use before creating runtime outputs')
    a=p.parse_args(); root=a.root.resolve()
    manifest=json.loads((root/'PUBLIC_ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
    expected={entry['path']:entry for entry in manifest['files']}
    assert len(expected)==len(manifest['files']), 'duplicate manifest path'
    for rel,entry in expected.items():
        q=(root/rel).resolve()
        assert q.is_relative_to(root), f'unsafe path: {rel}'
        data=q.read_bytes()
        assert len(data)==entry['bytes'], f'length mismatch: {rel}'
        assert hashlib.sha256(data).hexdigest()==entry['sha256'], f'hash mismatch: {rel}'
    sums={}
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        h,rel=line.split('  ',1)
        assert rel not in sums, f'duplicate checksum path: {rel}'
        q=(root/rel).resolve()
        assert q.is_relative_to(root), f'unsafe checksum path: {rel}'
        assert hashlib.sha256(q.read_bytes()).hexdigest()==h, f'checksum mismatch: {rel}'
        sums[rel]=h
    assert set(sums)==set(expected)|{'PUBLIC_ARTIFACT_MANIFEST.json'}
    if a.strict:
        actual={str(q.relative_to(root)).replace('\\','/') for q in root.rglob('*') if q.is_file()}
        assert actual==set(expected)|{'PUBLIC_ARTIFACT_MANIFEST.json','SHA256SUMS'}, sorted(actual^set(sums)^{'SHA256SUMS'})
    print(f'Verified {len(expected)} payload files and manifest checksums.')

if __name__=='__main__': main()

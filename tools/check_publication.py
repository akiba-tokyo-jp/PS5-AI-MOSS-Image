# SPDX-License-Identifier: GPL-3.0-or-later
"""Check staged public files and documentation; no console/network/inference."""
import hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/p for p in subprocess.check_output(['git','ls-files','--cached','-z'],cwd=ROOT).decode().split('\0') if p]
for p in files:
    relative=p.relative_to(ROOT).as_posix()
    assert not any(part in ('.local','secrets','vendor','toolchain','logs','__pycache__') for part in p.relative_to(ROOT).parts),relative
    assert '.private.' not in p.name and p.suffix not in ('.safetensors','.gguf','.prx','.elf','.bin','.a','.so'),relative
    assert p.stat().st_size<20<<20,relative
    if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'))
    if p.suffix=='.svg':ET.parse(p)
    if p.suffix in ('.md','.json','.py','.c','.cpp','.h','.inc','.ps1','.svg','.patch'):
        s=p.read_text(encoding='utf-8')
        if p.name!='check_publication.py':
            assert not re.search(r'https?://[^/\s]+\.sharepoint\.com|[A-Z]:[/\\]+Users[/\\]+|codex-remote-attachments|\.private\.json',s),relative
    if p.suffix=='.md':
        for target in re.findall(r'\[[^\]]*\]\(([^)\s]+)\)',p.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#'):assert (p.parent/target.split('#')[0]).exists(),(relative,target)
spec=json.loads((ROOT/'docs/openapi.json').read_text())
embedded=(ROOT/'native/gateway/openapi_schema.inc').read_text().splitlines()[2].rstrip(';')
assert json.loads(json.loads(embedded))==spec,'embedded schema differs'
for row in json.loads((ROOT/'docs/source-snapshot.json').read_text())['files']:
    assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['published_sha256'],row['file']
    indexed=subprocess.check_output(['git','show',':'+row['file']],cwd=ROOT)
    assert hashlib.sha256(indexed).hexdigest()==row['published_sha256'],('index hash',row['file'])
v=json.loads((ROOT/'docs/verification.json').read_text())
for test in v['tests']:
    image=ROOT/'docs/images'/('wai-1024.png' if test['size']=='1024x1024' else 'wai-256.png')
    assert hashlib.sha256(image.read_bytes()).hexdigest()==test['PNG_sha256']
print(json.dumps({'staged_files_checked':len(files),'source_hashes':'PASS','image_hashes':'PASS','relative_markdown_links':'PASS','json_svg':'PASS','embedded_schema':'PASS','public_file_policy':'PASS','inference_performed':False}))

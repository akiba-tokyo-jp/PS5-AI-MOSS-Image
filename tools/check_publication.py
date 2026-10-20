# SPDX-License-Identifier: GPL-3.0-or-later
"""Check the complete staged MOSS publication tree. No network or inference.

Findings print paths only: never matching text, credentials or their hashes.
The source snapshot records public checkout bytes, preserving explicit CRLF
attributes; the staged blobs are checked independently before publication.
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,posixpath,re,subprocess,sys,xml.etree.ElementTree as ET

MAX_FILE=20<<20
SNAPSHOT='docs/source-snapshot.json'
VERIFICATION='docs/verification.json'
FORBIDDEN_PARTS={'.local','secrets','vendor','toolchain','logs','__pycache__','.venv','node_modules'}
FORBIDDEN_SUFFIXES={'.safetensors','.gguf','.prx','.elf','.bin','.a','.so','.o','.obj','.dll','.lib','.log','.jsonl','.zip','.7z','.tar','.gz'}
TEXT_SUFFIXES={'.md','.json','.example','.py','.c','.cpp','.h','.hpp','.inc','.ps1','.svg','.patch','.txt','.toml','.yml','.yaml','.sh','.map','.ld'}
PRIVATE_TEXT=re.compile(r'https?://[^/\s]+\.sharepoint\.com|[A-Za-z]:[/\\]+Users[/\\]+|'+'codex-'+'remote-attachments',re.I)
SECRET_TEXT=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bBearer[ \t]+[A-Za-z0-9._~+/-]{24,}={0,2}|\b(?:api[_-]?token|password)[ \t]*[:=][ \t]*[\"\x27][A-Za-z0-9._~+/-]{32,}[\"\x27]',re.I)
LEGACY_NAME=re.compile(r'(?<![A-Za-z0-9_])(?:WAI|Anima)(?![A-Za-z0-9_])',re.I)

class Refused(Exception):
 def __init__(self,path):self.path=path

def need(ok,path):
 if not ok:raise Refused(path)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def safe_name(name):
 p=PurePosixPath(name)
 return bool(name)and not p.is_absolute()and '..'not in p.parts and '\\'not in name and all(x not in ('','.')for x in p.parts)
def json_value(raw,path):
 def unique(pairs):
  out={}
  for k,v in pairs:need(k not in out,path);out[k]=v
  return out
 try:return json.loads(raw.decode('utf-8-sig'),object_pairs_hook=unique,parse_constant=lambda _:need(False,path))
 except (ValueError,UnicodeError):raise Refused(path)
def git(root,*args,input=None):
 try:return subprocess.check_output(['git',*args],cwd=root,input=input,stderr=subprocess.DEVNULL)
 except subprocess.CalledProcessError:raise Refused('tools/check_publication.py')

def staged(root):
 rows=[]
 for entry in git(root,'ls-files','--stage','-z').split(b'\0'):
  if not entry:continue
  meta,name=entry.split(b'\t',1);mode,oid,stage=meta.split();name=name.decode('utf-8')
  need(safe_name(name)and mode in(b'100644',b'100755')and stage==b'0',name);rows.append((name,oid))
 need(bool(rows),'tools/check_publication.py')
 unique_oids=list(dict.fromkeys(oid for _,oid in rows));query=b'\n'.join(unique_oids)+b'\n'
 info=git(root,'cat-file','--batch-check',input=query).splitlines();need(len(info)==len(unique_oids),'tools/check_publication.py')
 sizes={}
 for oid,line in zip(unique_oids,info):
  fields=line.split();need(len(fields)==3 and fields[0]==oid and fields[1]==b'blob'and fields[2].isdigit(),'tools/check_publication.py');sizes[oid]=int(fields[2])
 for name,oid in rows:need(sizes[oid]<MAX_FILE,name)
 data=git(root,'cat-file','--batch',input=query);offset=0;blobs={}
 for oid in unique_oids:
  end=data.find(b'\n',offset);need(end>=offset,'tools/check_publication.py');header=data[offset:end].split();need(header==[oid,b'blob',str(sizes[oid]).encode()],'tools/check_publication.py');offset=end+1
  blobs[oid]=data[offset:offset+sizes[oid]];offset+=sizes[oid];need(data[offset:offset+1]==b'\n','tools/check_publication.py');offset+=1
 need(offset==len(data),'tools/check_publication.py')
 return {name:blobs[oid]for name,oid in rows}

def scan_file(name,raw):
 parts=PurePosixPath(name).parts;suffix=PurePosixPath(name).suffix.lower()
 need(not any(p.casefold()in FORBIDDEN_PARTS for p in parts)and '.private.'not in name.casefold()and PurePosixPath(name).name.casefold()not in('api-token','.env')and suffix not in FORBIDDEN_SUFFIXES,name)
 text=None
 if suffix in TEXT_SUFFIXES or name in('LICENSE','.gitignore','.gitattributes'):
  try:text=raw.decode('utf-8-sig')
  except UnicodeError:raise Refused(name)
  need(not PRIVATE_TEXT.search(text)and not SECRET_TEXT.search(text),name)
 if suffix=='.json'or name.endswith('.json.example'):json_value(raw,name)
 if suffix=='.svg':
  try:ET.fromstring(raw)
  except ET.ParseError:raise Refused(name)
 # Project narrative/examples must be MOSS-only. Third-party credit/licenses
 # and native compatibility identifiers are deliberately outside this rule.
 authored=(suffix=='.md'and not name.startswith('licenses/')and name!='THIRD_PARTY_NOTICES.md')or name in('docs/openapi.json','native/gateway/openapi_schema.inc','scripts/Invoke-MossOpenAI.ps1')
 if authored and text is not None:
  narrative=text
  if name=='docs/dashboard.md':
   # This exact required OFL derivative filename is a license link, not a
   # claim that this application implements another model's features.
   narrative=narrative.replace('](../licenses/WAI-Dashboard-Glyphs-OFL.txt)',']()')
  need(not LEGACY_NAME.search(narrative),name)
 return text

def inventory(root,files):
 need(SNAPSHOT in files,SNAPSHOT);value=json_value(files[SNAPSHOT],SNAPSHOT)
 need(value.get('schema')=='ps5-ai-moss-public-source-snapshot-v1'and value.get('model_id')=='moss-image-sft'and type(value.get('files'))is list,SNAPSHOT)
 seen=set()
 for row in value['files']:
  name=row.get('file')if type(row)is dict else None
  need(type(name)is str and safe_name(name)and name!=SNAPSHOT and name in files and name not in seen,SNAPSHOT);seen.add(name)
  need(type(row.get('published_bytes'))is int and row['published_bytes']>=0 and type(row.get('published_sha256'))is str and re.fullmatch('[0-9a-f]{64}',row['published_sha256']),name)
  checkout=root/name;need(checkout.is_file()and not checkout.is_symlink(),name);raw=checkout.read_bytes()
  need(len(raw)==row['published_bytes']and sha(raw)==row['published_sha256'],name)
  indexed=files[name]
  if indexed!=raw:
   attr=git(root,'check-attr','eol','--',name).decode().strip()
   need(attr.endswith(': eol: crlf')and b'\r'not in indexed and indexed.replace(b'\n',b'\r\n')==raw,name)
 need(seen==set(files)-{SNAPSHOT},SNAPSHOT)
 return len(seen)

def verification(files):
 need(VERIFICATION in files,VERIFICATION);v=json_value(files[VERIFICATION],VERIFICATION)
 need(v.get('schema')=='ps5-ai-moss-public-verification-v1'and v.get('model_id')=='moss-image-sft'and type(v.get('tests'))is list and len(v['tests'])>=2,VERIFICATION)
 latest=v.get('latest_results_document');need(type(latest)is str and latest.startswith('docs/moss-')and latest in files,VERIFICATION)
 ids=set();sizes=set()
 for test in v['tests']:
  need(type(test)is dict and type(test.get('id'))is str and test['id']not in ids and test.get('model_id')=='moss-image-sft',VERIFICATION);ids.add(test['id'])
  name=test.get('image');need(type(name)is str and safe_name(name)and name.startswith('docs/images/moss-')and name.endswith('.png')and name in files,VERIFICATION)
  need(type(test.get('PNG_bytes'))is int and test['PNG_bytes']==len(files[name])and test.get('PNG_sha256')==sha(files[name]),name)
  need(test.get('size')in('256x256','1024x1024')and type(test.get('native_generation_seconds'))in(int,float)and 0<test['native_generation_seconds']<86400 and type(test.get('profile'))is str and bool(test['profile']),VERIFICATION)
  raw=files[name];need(len(raw)>=24 and raw[:8]==b'\x89PNG\r\n\x1a\n'and raw[12:16]==b'IHDR',name)
  w=int.from_bytes(raw[16:20],'big');h=int.from_bytes(raw[20:24],'big');need(f'{w}x{h}'==test['size'],name);sizes.add(test['size'])
 need(sizes=={'256x256','1024x1024'},VERIFICATION)
 links=re.findall(r'!?\[[^\]]*\]\(([^)\s]+)\)',files[latest].decode('utf-8-sig'))
 linked={posixpath.normpath(str(PurePosixPath(latest).parent/target.split('#')[0]))for target in links if '://'not in target and not target.startswith('#')}
 need(all(test['image']in linked for test in v['tests']),latest)
 return len(v['tests'])

def check(root):
 root=Path(root).resolve();files=staged(root);texts={}
 for name,raw in files.items():texts[name]=scan_file(name,raw)
 for name,text in texts.items():
  if name.endswith('.md')and not name.startswith('licenses/')and text is not None:
   for target in re.findall(r'!?\[[^\]]*\]\(([^)\s]+)\)',text):
    if '://'in target or target.startswith('#'):continue
    target=target.split('#')[0].split('?')[0]
    # Resolve against the repository rather than accepting a local untracked
    # file as evidence of a publishable link.
    from urllib.parse import unquote
    p=(root/PurePosixPath(name).parent/unquote(target)).resolve()
    need(p.is_relative_to(root)and p.relative_to(root).as_posix()in files,name)
 spec=json_value(files.get('docs/openapi.json',b''),'docs/openapi.json')
 try:
  lines=files['native/gateway/openapi_schema.inc'].decode('utf-8-sig').splitlines();embedded=json.loads(json.loads(lines[2].rstrip(';')))
 except (KeyError,ValueError,IndexError,UnicodeError):raise Refused('native/gateway/openapi_schema.inc')
 need(embedded==spec,'native/gateway/openapi_schema.inc')
 need(spec.get('info',{}).get('version')=='0.2.0-turbo4-verification'and spec.get('x-runtime-profile',{}).get('moss_operating_profile')is False and spec['x-runtime-profile'].get('max_generation_attempts_per_native_process')==4 and spec['x-runtime-profile'].get('production_runtime_admitted')is False,'docs/openapi.json')
 count=inventory(root,files);images=verification(files)
 return dict(staged_files_checked=len(files),MOSS_source_snapshot_files=count,MOSS_benchmark_images=images,public_file_policy='PASS',authored_markdown_links='PASS',embedded_schema='PASS',findings_reveal_paths_only=True,inference_performed=False)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
 try:result=check(a.root)
 except Refused as error:print(error.path,file=sys.stderr);return 1
 except Exception:print('tools/check_publication.py',file=sys.stderr);return 1
 print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())

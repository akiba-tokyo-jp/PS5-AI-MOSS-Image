# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the ordinary Turbo4 API service using reviewed, preconfigured dependencies.

No PS5 connection/deployment, credential generation or model download. Prepare
the dependency config and source freeze as documented before calling this tool.
"""
from pathlib import Path
import argparse,subprocess,sys,shutil,json,hashlib
from moss_feature_policy import feature_manifest_fields,ffn_manifest_fields
from moss_dit_policy import fields as dit_fields
from moss_kernel_policy import fields as kernel_fields
from moss_lora_precision_policy import fields as lora_fields
R=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--source-freeze',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--jobs',type=int,default=2);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.out.resolve()
    if out.exists()or out.is_relative_to(R):raise RuntimeError('Use a fresh build directory outside the source checkout')
    if not 1<=a.jobs<=8:raise RuntimeError('jobs must be1..8')
    out.mkdir(parents=True)
    def run(name,args):subprocess.run([sys.executable,'-B',str(R/'tools'/name),'--config',str(a.config),*map(str,args)],check=True)
    # Generate the schema before compiling/fixing consumed source bytes.
    run('build_gateway.py',['--out',out/'gateway','--native','--moss-turbo-trial','--moss-img2img-trial','--service-mode'])
    freeze=a.source_freeze
    if freeze is None:
        config=json.loads(a.config.read_text());value=json.loads((R/'config/moss-source-freeze.json.example').read_text())
        value.update(frozen=True,repository_root=R.as_posix(),private_vendor_root=Path(config['SD']).resolve().as_posix(),sources={},new_core_compilation_required=True,old_core_reuse_allowed=False,scope='Local consumed-source freeze; fresh compilation required; not a hardware success receipt')
        value.update(feature_manifest_fields(False,True,True));value.update(ffn_manifest_fields(False,0));value.update(dit_fields(True));value.update(kernel_fields(True));value.update(lora_fields(None,core=True))
        names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=R).decode().split('\0')
        for name in sorted(set(filter(None,names))):
            path=(R/name).resolve()
            if not path.is_relative_to(R) or not path.is_file():raise RuntimeError('Invalid source input')
            raw=path.read_bytes();value['sources'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        freeze=out/'source-freeze.private.json';freeze.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    features=['--moss-f32-accumulation','--moss-turbo-trial','--moss-bounded-weight-upload-trial']
    run('build_moss_core.py',['--out',out/'core','--source-freeze',freeze,'--jobs',a.jobs,*features,'--moss-dit-optimizations','--moss-dit-kernels',*(['--check-only']if a.prepare_only else[])])
    if a.prepare_only:
        print('Prepared gateway, local source freeze and75-object core plan. Native application not built; no console contacted.');return
    run('build.py',['--out',out/'native','--core-manifest',out/'core/manifest.private.json','--experimental-small-trial','--trial-10g','--performance-trial-variant','1','--precision','fp16',*features,'--moss-dit-optimization-mask','15','--moss-dit-kernel-mask','15','--moss-fa-shmem-no-limit','--moss-img2img-trial','--service-mode'])
    shutil.copyfile(R/'config/moss-service.json.example',out/'moss-service.json.example')
    print('Service build complete. No console was contacted. See docs/deployment.md for installation and credentials.')
if __name__=='__main__':main()

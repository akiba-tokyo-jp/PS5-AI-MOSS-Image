# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch pinned public sources and apply explicit overlays. Local development only."""
import argparse, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();base=a.out.resolve();base.mkdir(parents=True,exist_ok=True)
    lock=json.loads((ROOT/'dependencies.lock.json').read_text())
    for repo in lock['source_repositories']:
        dest=base/repo['directory']
        if not dest.exists():subprocess.run(['git','clone','--no-checkout',repo['url'],str(dest)],check=True)
        elif subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True).strip():raise RuntimeError('Keep modified checkout intact: '+str(dest))
        subprocess.run(['git','-C',str(dest),'fetch','origin',repo['revision']],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',repo['revision']],check=True)
        subprocess.run(['git','-C',str(dest),'submodule','update','--init','--recursive'],check=True)
    for name,checkout in [('stable-diffusion','stable-diffusion.cpp'),('mesa','PS5_Mesa')]:
        overlay=ROOT/'patches'/name/'overlay'
        if overlay.exists():
            for file in overlay.rglob('*'):
                if file.is_file():target=base/checkout/file.relative_to(overlay);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    print('Pinned sources prepared. SDK installation and dependency compilation are separate; see docs/build.md.')
if __name__=='__main__':main()

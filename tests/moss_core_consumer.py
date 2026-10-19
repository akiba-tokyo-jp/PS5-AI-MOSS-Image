# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only real-core admission and negative manifest checks. No PS5/inference."""
import argparse,copy,importlib.util,json,tempfile
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();r=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('moss_build',r/'tools/build.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=json.loads(a.config.read_text());manifest=json.loads(a.manifest.read_text())
baseline=m.archive_members(Path(c['DEPS'])/'libstable-diffusion.graph-admission-1024-v3.a')
a.out.mkdir(parents=True,exist_ok=True)
m.validate_moss_core(a.manifest,baseline,c)
checks=['completed_same_checkout_accepted']
def rejects(path,config,label):
    try:m.validate_moss_core(path,baseline,config)
    except RuntimeError:checks.append(label);return
    raise AssertionError('Unexpected acceptance: '+label)
other=copy.deepcopy(c);other['SD']=str(a.out/'different-SD-checkout')
rejects(a.manifest,other,'different_SD_checkout_rejected')
with tempfile.TemporaryDirectory(dir=a.out) as directory:
    file=Path(directory)/'fixture.json'
    for key,value,label in [('state','building','unfinished_core_rejected'),
                            ('old_archive_members_reused',1,'reused_old_member_rejected'),
                            ('MOSS_RUNTIME_ADMITTED',1,'enabled_runtime_rejected')]:
        bad=copy.deepcopy(manifest);bad[key]=value
        file.write_text(json.dumps(bad));rejects(file,c,label)
result={'success':True,'checks':checks,'PS5_contacted':False,'inference_executed':False}
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

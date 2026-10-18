"""Check V17 mask source/ABI/build export; no model, GPU or console use."""
import ast,hashlib,json,struct,subprocess,os
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 out=Path(os.environ['PS5_WAI_MASK_TEST_OUT']);host=out/'host';assert host.is_dir()
 canonical=(R/'native/app/engine.h').read_bytes()
 assert canonical==(R/'native/gateway/engine.h').read_bytes(),'Native/gateway ABI mismatch'
 for name in ['mask_image_id','mask_image_path','mask_image_sha256','mask_denoise_evaluations','mask_prepared','mask_preserved_pixels','mask_changed_pixels']:
  assert name in canonical.decode(),name
 p=json.loads((R/'build-profiles/native.json').read_text());assert p['controlnet_batch_pair']==1
 recipes=[p['runner_compile'],*p['reference_compile']]
 sources=[x[x.index('-c')+1] for x in recipes]
 assert len(recipes)==7 and any('/overlay/src/pipeline/image.cpp' in x for x in sources)
 assert any('/overlay/src/pipeline/diffusion_engine.cpp' in x for x in sources)
 c=json.loads((R/'config/native.json.example').read_text())['engine']
 assert c['gpu_pool_bytes']==9422<<20 and c['max_vram_gib']=='Vulkan0=8.75'
 assert c['working_reserve_bytes']==1<<30
 spec=json.loads((R/'docs/openapi.json').read_text())
 embedded=(R/'native/gateway/openapi_schema.inc').read_text().splitlines()[2].rstrip(';')
 assert json.loads(json.loads(embedded))==spec
 assert 'mask_image_id' in spec['components']['schemas']['ImageRequest']['properties']
 assert 'mask' in spec['components']['schemas']['ImageEditRequest']['properties']
 for f in (R/'tests').glob('mask_*.py'):ast.parse(f.read_text())
 for row in json.loads((R/'docs/source-snapshot.json').read_text())['files']:
  assert hashlib.sha256((R/row['file']).read_bytes()).hexdigest()==row['published_sha256'],row['file']
 fixtures=host/'owner-data';fixtures.mkdir(exist_ok=False)
 payloads=(b'PS5MSK1\0'+struct.pack('<II',8,8)+bytes(63),b'PS5MSK1\0'+struct.pack('<II',8,8)+bytes(65),b'BADMSK1\0'+struct.pack('<II',8,8)+bytes(64),b'PS5MSK1\0'+struct.pack('<II',0,8))
 for i,data in enumerate(payloads):(fixtures/f'bad{i}.mask').write_bytes(data)
 for command in ([host/'mask-composite.exe'],[host/'mask-native-owner.exe',fixtures],[host/'mask-completion-contract.exe']):
  q=subprocess.run(list(map(str,command)),capture_output=True,text=True,errors='replace');assert q.returncode==0,q.stdout+q.stderr;print(q.stdout.strip())
 result=dict(state='passed',native_gateway_ABI_identical=True,core_recipe_count=7,controlnet_batch_pair=1,Direct_pool_bytes=9422<<20,PS5_connections=0,model_inference=False)
 (host/'export-profile-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

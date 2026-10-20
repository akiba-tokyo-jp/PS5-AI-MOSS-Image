"""Arm a NEW local Turbo config only after a verified offline composition.

No model inference, PS5 access, original config replacement or base model read.
The default template stays unarmed; no hash or successful preparation is invented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compose_moss_turbo_lora as c


def arm(template,composition_receipt,output):
    template,composition_receipt=Path(template).resolve(strict=True),Path(composition_receipt).resolve(strict=True)
    output=Path(output).resolve()
    repository=Path(__file__).resolve().parents[2]
    if output.exists() or output.is_relative_to(repository) or output in (template,composition_receipt):
        raise ValueError('Preserve source template/configs; use a fresh output outside repository')
    record=json.loads(composition_receipt.read_text())
    required={'schema':c.SCHEMA,'rank':144,'matrix_count':504,'matrix_payload_bytes':1337720832,
        'full_coefficient_reread':True,'actual_turbo_alphas_verified':252,'runtime_multiplier':1.0,
        'base_checkpoint_opened':False,'base_checkpoint_merged_or_requantized':False,
        'tensor_precision':'F32','PS5_access':False,'model_inference':False}
    if any(type(record.get(k))is not type(v) or record.get(k)!=v for k,v in required.items()):
        raise ValueError('Actual complete composition receipt required; no placeholder preparation')
    inputs=record.get('inputs')
    if not isinstance(inputs,list) or len(inputs)!=2 or inputs[0].get('sha256')!=c.NSFW_F32_SHA256 or inputs[0].get('bytes')!=c.NSFW_F32_BYTES or inputs[1].get('sha256')!=c.TURBO_SHA256 or inputs[1].get('bytes')!=c.TURBO_BYTES:
        raise ValueError('Exact selected NSFW and pinned Turbo input identities required')
    artifact=record['output'];path=Path(artifact['path']).resolve(strict=True)
    if path.name!=c.OUTPUT_NAME or path.stat().st_size!=artifact['bytes']:
        raise ValueError('Fresh composed output is missing or has wrong identity')
    with path.open('rb')as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=artifact['sha256']:
            raise ValueError('Composed coefficient bytes changed')
        stream.seek(0);header,start,_=c.read_header(stream)
    metadata=header.get('__metadata__',{})
    for key,value in {'ps5_ai_moss_lora_schema':c.SCHEMA,'rank':'144','runtime_multiplier':'1.0',
        'nsfw_derivative_sha256':c.NSFW_F32_SHA256,'turbo_sha256':c.TURBO_SHA256}.items():
        if metadata.get(key)!=value:raise ValueError('Actual composed-header provenance differs')
    expected={}
    for _,_,side,ins,outs,name,_ in c.catalog():expected[name]=[144,ins]if side=='down'else[outs,144]
    c.validate_rows(header,expected,artifact['bytes']-start)
    document=json.loads(template.read_text());engine=document['engine']
    if engine.get('moss_lora_path')!='/app0/models/'+c.OUTPUT_NAME or engine.get('moss_lora_strength')!=1.0 or engine.get('moss_lora_sha256')is not None:
        raise ValueError('Exact unarmed composed-adapter template required')
    if engine.get('gpu_pool_bytes')!=10737418240 or engine.get('working_reserve_bytes')!=1073741824:
        raise ValueError('Do not change the reviewed memory reservation during config arming')
    engine['moss_lora_sha256']=artifact['sha256']
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8')as stream:json.dump(document,stream,indent=2);stream.write('\n')
    return dict(path=str(output),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                composed_artifact=artifact,PS5_access=False,model_inference=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--composition-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(arm(a.template,a.composition_receipt,a.output),indent=2))


if __name__=='__main__':main()

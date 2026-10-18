"""Host-only mask/ControlNet ownership tests; the engine always fails."""
import concurrent.futures
import io,json,os,secrets,socket,subprocess,time
from pathlib import Path
from PIL import Image
import mask_api_protocol as module
O=Path(os.environ['PS5_WAI_MASK_TEST_OUT'])
def png(mode,size=(256,256),color=None):
    image=Image.new(mode,size,color);out=io.BytesIO();image.save(out,format='PNG');return out.getvalue()
def extra_protocol(module):
    checks = []
    def check(value, name):
        assert value, name
        checks.append(name)
    # All roots/credentials below are fresh local fixtures, never production.
    root = O / 'host/combined-extra'
    assert not root.exists()
    native, gateway = root / 'native', root / 'gateway'
    token = secrets.token_hex(32)
    for p in (native, gateway):
        (p / 'config').mkdir(parents=True)
        (p / 'config/api-token').write_text(token, encoding='ascii')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    processes = []
    try:
        for exe, folder, p in [('native-failure.exe', native, 18188), ('gateway-forwarder.exe', gateway, port)]:
            process = subprocess.Popen([str(O / 'host' / exe), '--root', str(folder), '--bind', '127.0.0.1', '--port', str(p)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(process)
            module.wait_health(p, token, process)
        call = lambda p, method, path, data=b'', kind='application/json': module.request(p, token, method, path, data, kind)
        source = png('RGB', color=(90, 140, 180))
        reference = png('RGB', color=(180, 100, 130))
        control = png('RGB', color=(0, 0, 0))
        mask_image = Image.new('RGBA', (256, 256), (0, 0, 0, 255))
        mask_image.paste((0, 0, 0, 0), (0, 0, 128, 256))
        stream = io.BytesIO()
        mask_image.save(stream, format='PNG')
        alpha = stream.getvalue()
        code, ctrl = call(18188, 'POST', '/api/inputs', control, 'image/png')
        check(code == 201, 'persistent prepared control map upload')
        fields = [('prompt', 'host mask controlnet failure fixture'), ('size', '256x256'),
            ('reference_mode', 'combined'), ('controlnet', 'openpose'),
            ('control_image_id', ctrl['image_id']), ('control_strength', '0.7'), ('denoise_strength', '0.8')]
        body, kind = module.form(fields, [('image', 'image/png', source),
            ('reference_image', 'image/png', reference), ('mask', 'image/png', alpha)])
        before = set((native / 'jobs').glob('*.request.json'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(call, port, 'POST', '/v1/images/edits', body, kind)
            deadline = time.monotonic() + 5
            requests = []
            while time.monotonic() < deadline:
                requests = [p for p in (native / 'jobs').glob('*.request.json') if p not in before]
                if requests:
                    break
                time.sleep(.01)
            check(len(requests) == 1, 'combined multipart admits exactly one upstream job')
            forwarded = json.loads(requests[0].read_text())
            check(forwarded['controlnet'] == 'openpose' and forwarded['control_strength'] == .7,
                'multipart control strength forwarded as numeric, not quoted text')
            check(forwarded['control_image_id'] == ctrl['image_id'], 'existing control ID passed unchanged')
            ids = [forwarded[k] for k in ('init_image_id', 'reference_image_id', 'mask_image_id', 'control_image_id')]
            check(len(set(ids)) == 4, 'four distinct image roles resolved')
            for i, image_id in enumerate(ids):
                code, metadata = call(18188, 'GET', '/api/inputs/' + image_id)
                check(code == 200 and metadata['references'] == 1, 'active four-role lease ' + str(i))
                check(call(18188, 'DELETE', '/api/inputs/' + image_id)[0] == 409,
                    'active four-role deletion refused ' + str(i))
            code, result = future.result(timeout=15)
        check(code == 500 and 'Native job failed' in result['error']['message'],
            'combined mask+IP+ControlNet propagates intentional failure without PNG')
        job = json.loads(requests[0].with_name(requests[0].name.replace('.request.json', '.json')).read_text())
        check(len(job) == 36 and job['inpainting'] and job['controlnet'] == 'openpose',
            'actual native36-field job preserves mask and ControlNet provenance')
        check(not job['gpu_used'] and not job['mask_prepared'] and not job['gpu_controlnet_verified'],
            'failure fixture never manufactures GPU, mask or ControlNet success')
        check(call(18188, 'GET', '/api/inputs/' + ctrl['image_id'])[1]['references'] == 0,
            'native control lease returned to zero')
        check(len(list((native / 'inputs').glob('*.rgb'))) == 1 and not list((native / 'inputs').glob('*.mask')),
            'owned source/reference/mask removed; caller control map retained')
        check(call(18188, 'GET', '/api/inputs/' + ctrl['image_id'])[0] == 200,
            'caller-owned control survives gateway transient cleanup')
        before_count = len(list((native / 'jobs').glob('*.request.json')))
        for bad in ['0', '2', 'NaN', '"0.7"']:
            wrong = [(k, bad if k == 'control_strength' else v) for k, v in fields]
            data, typ = module.form(wrong, [('image', 'image/png', source), ('mask', 'image/png', alpha)])
            check(call(port, 'POST', '/v1/images/edits', data, typ)[0] == 400,
                'invalid multipart control strength rejected ' + bad)
        data, typ = module.form(fields, [('image', 'image/png', source), ('control_image', 'image/png', control)])
        check(call(port, 'POST', '/v1/images/edits', data, typ)[0] == 400,
            'control_image file remains unsupported')
        check(len(list((native / 'jobs').glob('*.request.json'))) == before_count,
            'invalid combined edits send zero upstream generation POSTs')
        check(call(18188, 'GET', '/api/inputs/' + ctrl['image_id'])[0] == 200,
            'invalid edits preserve caller control asset')
        check(call(18188, 'DELETE', '/api/inputs/' + ctrl['image_id'])[0] == 200, 'caller can delete unused control asset')
        check(not list((native / 'output').glob('*.png')) and not list((gateway / 'output').glob('*.png')),
            'zero generated PNGs from combined host regressions')
    finally:
        for process, p in reversed(list(zip(processes, (18188, port)))):
            if process.poll() is None:
                try:
                    module.request(p, token, 'POST', '/api/admin/shutdown')
                    process.communicate(timeout=5)
                except Exception:
                    process.terminate()
                    process.communicate(timeout=5)
    return checks

if __name__=='__main__':
    checks=extra_protocol(module)
    result=dict(state='passed',checks=checks,checks_passed=len(checks),
        PS5_connections=0,Windows_inference=False,generated_PNGs=0,
        backend='Intentional host failure fixture only')
    (O/'host/mask-controlnet-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

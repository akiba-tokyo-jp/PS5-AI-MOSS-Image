param(
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$Prompt,
    [ValidateSet('img2img','ip_adapter','combined','inpaint')][string]$Mode='img2img',
    [string]$ReferenceImage,
    [string]$Mask,
    [ValidateSet('alpha','grayscale')][string]$MaskMode='alpha',
    [string]$NegativePrompt='',
    [ValidatePattern('^[0-9]+x[0-9]+$')][string]$Size='256x256',
    [ValidateRange(1,100)][int]$Steps=20,
    [ValidateRange(0,30)][double]$CfgScale=5,
    [long]$Seed=20261018,
    [ValidateSet('euler','euler_a','dpm++2m')][string]$Sampler='euler_a',
    [ValidateRange(0.01,1)][double]$DenoiseStrength=0.6,
    [ValidateRange(0.01,1.5)][double]$IPAdapterStrength=0.7,
    [ValidateRange(0,1.5)][double]$MangaStrength=1,
    [switch]$MangaMonochrome,
    [string]$BaseUrl='http://192.168.1.100:8189/v1',
    [string]$TokenFile=(Join-Path $PSScriptRoot '..\secrets\api-token'),
    [string]$OutputDirectory=(Join-Path (Get-Location) 'reference-output')
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Net.Http
$base=[Uri]$BaseUrl
$address=$null
if (-not [Net.IPAddress]::TryParse($base.Host,[ref]$address) -or $address.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'Use an explicit trusted LAN IPv4 address.' }
$octets=$address.GetAddressBytes()
if ($base.Scheme -ne 'http' -or $base.UserInfo -or $base.Query -or $base.Fragment -or -not ($octets[0] -eq 10 -or $octets[0] -eq 127 -or ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31) -or ($octets[0] -eq 192 -and $octets[1] -eq 168))) { throw 'Use a trusted LAN HTTP endpoint with no embedded credentials.' }
if ($Seed -lt -1) { throw 'Seed must be -1 or nonnegative.' }
if ($ReferenceImage -and $Mode -ne 'combined') { throw 'A separate ReferenceImage requires combined mode.' }
if ($Mode -eq 'inpaint' -and -not $Mask) { throw 'Inpaint mode requires a PNG Mask.' }
if ($Mask -and $Mode -eq 'ip_adapter') { throw 'Mask requires img2img, inpaint or combined mode.' }
if (-not $Mask -and $PSBoundParameters.ContainsKey('MaskMode')) { throw 'MaskMode requires a Mask.' }
$dims=$Size.Split('x')
foreach ($dim in $dims) { if ([long]$dim -lt 64 -or [long]$dim -gt 1024 -or [long]$dim % 64) { throw 'Dimensions must be64..1024 and multiples of64.' } }
$secret=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $TokenFile).Path).Trim()
if ($secret.Length -lt 32 -or $secret.Length -gt 256 -or $secret -match '[^!-~]') { throw 'Invalid local credential file.' }
$handler=New-Object Net.Http.HttpClientHandler
$handler.AllowAutoRedirect=$false
$handler.UseProxy=$false
$client=New-Object Net.Http.HttpClient($handler)
$client.Timeout=[TimeSpan]::FromMinutes(30)
$client.DefaultRequestHeaders.ExpectContinue=$false
$client.DefaultRequestHeaders.Authorization=New-Object Net.Http.Headers.AuthenticationHeaderValue('Bearer',$secret)
$form=New-Object Net.Http.MultipartFormDataContent
$response=$null
try {
    $culture=[Globalization.CultureInfo]::InvariantCulture
    $fields=[ordered]@{ model='wai-illustrious-sdxl-v17.0';prompt=$Prompt;negative_prompt=$NegativePrompt;size=$Size;response_format='b64_json';reference_mode=$Mode;steps=$Steps.ToString($culture);cfg_scale=$CfgScale.ToString('R',$culture);seed=$Seed.ToString($culture);sampler=$Sampler;manga_strength=$MangaStrength.ToString('R',$culture) }
    if ($Mode -ne 'ip_adapter') { $fields.denoise_strength=$DenoiseStrength.ToString('R',$culture) }
    if ($Mode -eq 'ip_adapter' -or $Mode -eq 'combined') { $fields.ip_adapter_strength=$IPAdapterStrength.ToString('R',$culture) }
    $fields.manga_monochrome=$MangaMonochrome.IsPresent.ToString().ToLowerInvariant()
    if ($Mask) { $fields.mask_mode=$MaskMode }
    foreach ($key in $fields.Keys) { $form.Add((New-Object Net.Http.StringContent([string]$fields[$key])),$key) }
    $inputs=[ordered]@{ image=$Image }
    if ($ReferenceImage) { $inputs.reference_image=$ReferenceImage }
    if ($Mask) { $inputs.mask=$Mask }
    foreach ($key in $inputs.Keys) {
        $file=Get-Item -LiteralPath $inputs[$key]
        if ($file.PSIsContainer -or $file.Length -le 0 -or $file.Length -gt 8MB) { throw 'Each input must be a PNG/JPEG file of1..8MiB.' }
        $type=switch ($file.Extension.ToLowerInvariant()) { '.png' {'image/png'} '.jpg' {'image/jpeg'} '.jpeg' {'image/jpeg'} default { throw 'Use PNG or JPEG.' } }
        if ($key -eq 'mask' -and ($type -ne 'image/png' -or $file.Length -ge 4MB)) { throw 'Mask must be PNG below4MiB.' }
        $stream=[IO.File]::OpenRead($file.FullName)
        $content=New-Object Net.Http.StreamContent($stream)
        $content.Headers.ContentType=New-Object Net.Http.Headers.MediaTypeHeaderValue($type)
        $form.Add($content,$key,$file.Name)
    }
    $timer=[Diagnostics.Stopwatch]::StartNew()
    # Exactly one generation POST; no automatic retry after a timeout/disconnect.
    $response=$client.PostAsync($BaseUrl.TrimEnd('/')+'/images/edits',$form).GetAwaiter().GetResult()
    $raw=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $timer.Stop()
    if (-not $response.IsSuccessStatusCode) { throw ('PS5 API HTTP'+[int]$response.StatusCode+': '+$raw) }
    $result=$raw | ConvertFrom-Json
    if ($result.data.Count -ne 1 -or -not $result.data[0].b64_json) { throw 'Expected one base64 PNG.' }
    $bytes=[Convert]::FromBase64String($result.data[0].b64_json)
    if ($bytes.Length -lt 24 -or $bytes.Length -gt 32MB -or [BitConverter]::ToString($bytes,0,8) -ne '89-50-4E-47-0D-0A-1A-0A') { throw 'Invalid PNG response.' }
    $nativeId=@($response.Headers.GetValues('X-PS5-Native-Job-ID'))[0]
    if ($nativeId -notmatch '^[0-9a-f]{32}$') { throw 'Missing native job ID.' }
    $nativeBase=New-Object UriBuilder($base);$nativeBase.Port=8188;$nativeBase.Path='/api/jobs/'+$nativeId
    $job=Invoke-RestMethod -Uri $nativeBase.Uri -Headers @{Authorization='Bearer '+$secret} -TimeoutSec 15
    if ($job.status -ne 'completed' -or -not $job.gpu_used) { throw 'Native job is not GPU-verified completed.' }
    if ($Mask -and (-not $job.inpainting -or -not $job.mask_prepared -or $job.mask_image_sha256 -notmatch '^[0-9a-f]{64}$')) { throw 'Masked pipeline/provenance was not verified.' }
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $output=Join-Path $OutputDirectory ($nativeId+'.png')
    $save=[IO.File]::Open($output,[IO.FileMode]::CreateNew)
    try { $save.Write($bytes,0,$bytes.Length) } finally { $save.Dispose() }
    $receipt=[ordered]@{path=$output;mode=$Mode;native_job_id=$nativeId;generation_seconds=$job.generation_seconds;windows_api_seconds=$timer.Elapsed.TotalSeconds;actual_steps=$job.actual_steps;ip_adapter_tokens=$job.ip_adapter_tokens;gpu_ip_attention_verified=$job.gpu_ip_attention_verified;sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}
    if ($Mask) {
        $receipt.inpainting=$job.inpainting
        $receipt.mask_image_sha256=$job.mask_image_sha256
        $receipt.mask_denoise_evaluations=$job.mask_denoise_evaluations
        $receipt.mask_preserved_pixels=$job.mask_preserved_pixels
        $receipt.mask_changed_pixels=$job.mask_changed_pixels
    }
    [IO.File]::WriteAllText($output+'.json',($receipt | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    [PSCustomObject]$receipt
} catch {
    throw ('Image request failed. Do not automatically resend; check8188 /health and the known job first. '+$_.Exception.Message)
} finally {
    if ($response) { $response.Dispose() }
    $form.Dispose();$client.Dispose();$handler.Dispose();$secret=$null
}

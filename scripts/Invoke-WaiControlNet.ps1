param(
    [Parameter(Mandatory=$true)][ValidateSet('canny','openpose')][string]$ControlNet,
    [Parameter(Mandatory=$true)][string]$ControlImage,
    [Parameter(Mandatory=$true)][string]$Prompt,
    [string]$NegativePrompt='low quality, blurry, text, watermark',
    [ValidateRange(0.01,1.5)][double]$ControlStrength=1,
    [ValidateRange(0,1.5)][double]$MangaStrength=1,
    [ValidateRange(1,100)][int]$Steps=20,
    [ValidateRange(0,30)][double]$CfgScale=5,
    [long]$Seed=20261018,
    [ValidateSet('euler','euler_a','dpm++2m')][string]$Sampler='euler_a',
    [string]$BaseUrl='http://192.168.1.100:8189/v1',
    [string]$NativeBaseUrl='http://192.168.1.100:8188',
    [string]$TokenFile=(Join-Path $PSScriptRoot '..\secrets\api-token'),
    [string]$OutputDirectory='D:\PS5-AI\controlnet-output'
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Net.Http
function Test-LanUri([string]$Text) {
    $u=[Uri]$Text;$a=$null
    if (-not [Net.IPAddress]::TryParse($u.Host,[ref]$a) -or $a.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'Use a trusted LAN IPv4 address.' }
    $b=$a.GetAddressBytes()
    if ($u.Scheme -ne 'http' -or $u.UserInfo -or $u.Query -or $u.Fragment -or -not ($b[0] -eq 10 -or $b[0] -eq 127 -or ($b[0] -eq 172 -and $b[1] -ge 16 -and $b[1] -le 31) -or ($b[0] -eq 192 -and $b[1] -eq 168))) { throw 'Use a trusted LAN HTTP endpoint without embedded credentials.' }
    return $u
}
$gateway=Test-LanUri $BaseUrl;$native=Test-LanUri $NativeBaseUrl
if ($gateway.Host -ne $native.Host) { throw 'Native API and OpenAI gateway must be on the same PS5.' }
if ($Seed -lt -1) { throw 'Seed must be -1 or nonnegative.' }
foreach ($value in @($ControlStrength,$MangaStrength,$CfgScale)) { if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { throw 'Numeric parameters must be finite.' } }
foreach ($text in @($Prompt,$NegativePrompt)) { if ([Text.Encoding]::UTF8.GetByteCount($text) -gt 4096) { throw 'Prompt fields must fit4096 UTF-8 bytes.' } }
$file=Get-Item -LiteralPath $ControlImage
if ($file.PSIsContainer -or $file.Length -lt 24 -or $file.Length -gt 8MB) { throw 'Use one prepared RGB edge/pose PNG below8MiB.' }
$inputBytes=[IO.File]::ReadAllBytes($file.FullName)
if ([BitConverter]::ToString($inputBytes,0,8) -ne '89-50-4E-47-0D-0A-1A-0A') { throw 'Prepared control map must be PNG.' }
$width=0L;$height=0L
foreach ($i in 16..19) { $width=$width*256+$inputBytes[$i] }
foreach ($i in 20..23) { $height=$height*256+$inputBytes[$i] }
foreach ($d in @($width,$height)) { if ($d -lt 64 -or $d -gt 1024 -or $d%64) { throw 'Control-map dimensions must be64..1024, multiples of64.' } }
$secret=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $TokenFile).Path).Trim()
if ($secret.Length -lt 32 -or $secret.Length -gt 256 -or $secret -match '[^!-~]') { throw 'Invalid local credential file.' }
$handler=New-Object Net.Http.HttpClientHandler;$handler.AllowAutoRedirect=$false;$handler.UseProxy=$false
$client=New-Object Net.Http.HttpClient($handler);$client.Timeout=[TimeSpan]::FromMinutes(30)
$client.DefaultRequestHeaders.ExpectContinue=$false
$client.DefaultRequestHeaders.Authorization=New-Object Net.Http.Headers.AuthenticationHeaderValue('Bearer',$secret)
$inputId=$null;$jobConfirmedTerminal=$false;$response=$null;$content=$null
try {
    foreach ($endpoint in @($NativeBaseUrl,$BaseUrl.TrimEnd('/').Substring(0,$BaseUrl.TrimEnd('/').LastIndexOf('/')))) {
        $h=$client.GetStringAsync($endpoint.TrimEnd('/')+'/health').GetAwaiter().GetResult() | ConvertFrom-Json
        if (-not $h.engine_available -or -not $h.gpu_available -or $h.running_jobs -or $h.queued_jobs) { throw 'PS5 backend is unavailable or busy; no generation submitted.' }
    }
    $content=New-Object Net.Http.ByteArrayContent(,$inputBytes)
    $content.Headers.ContentType=New-Object Net.Http.Headers.MediaTypeHeaderValue('image/png')
    $response=$client.PostAsync($NativeBaseUrl.TrimEnd('/')+'/api/inputs',$content).GetAwaiter().GetResult()
    $raw=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if ([int]$response.StatusCode -ne 201) { throw ('Control-map upload unconfirmed: HTTP'+[int]$response.StatusCode) }
    $uploaded=$raw | ConvertFrom-Json;$inputId=$uploaded.image_id
    if ($inputId -notmatch '^[0-9a-f]{32}$' -or $uploaded.width -ne $width -or $uploaded.height -ne $height) { throw 'Invalid uploaded control-map identity or dimensions.' }
    $response.Dispose();$response=$null;$content.Dispose();$content=$null
    $request=[ordered]@{model='wai-illustrious-sdxl-v17.0';prompt=$Prompt;negative_prompt=$NegativePrompt;size=($width.ToString()+'x'+$height.ToString());n=1;response_format='b64_json';steps=$Steps;cfg_scale=$CfgScale;seed=$Seed;sampler=$Sampler;manga_strength=$MangaStrength;manga_monochrome=$false;controlnet=$ControlNet;control_image_id=$inputId;control_strength=$ControlStrength}
    $content=New-Object Net.Http.StringContent(($request | ConvertTo-Json -Compress),[Text.Encoding]::UTF8,'application/json')
    $timer=[Diagnostics.Stopwatch]::StartNew()
    # One generation POST only. A timeout/disconnect is never automatically retried.
    $response=$client.PostAsync($BaseUrl.TrimEnd('/')+'/images/generations',$content).GetAwaiter().GetResult()
    $raw=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult();$timer.Stop()
    if (-not $response.IsSuccessStatusCode) { throw ('PS5 API HTTP'+[int]$response.StatusCode+': '+$raw) }
    $result=$raw | ConvertFrom-Json
    if ($result.data.Count -ne 1 -or -not $result.data[0].b64_json) { throw 'Expected one base64 PNG.' }
    $bytes=[Convert]::FromBase64String($result.data[0].b64_json)
    if ($bytes.Length -lt 24 -or $bytes.Length -gt 32MB -or [BitConverter]::ToString($bytes,0,8) -ne '89-50-4E-47-0D-0A-1A-0A') { throw 'Invalid PNG response.' }
    $nativeId=@($response.Headers.GetValues('X-PS5-Native-Job-ID'))[0]
    if ($nativeId -notmatch '^[0-9a-f]{32}$') { throw 'Missing native job ID.' }
    $job=$client.GetStringAsync($NativeBaseUrl.TrimEnd('/')+'/api/jobs/'+$nativeId).GetAwaiter().GetResult() | ConvertFrom-Json
    $jobConfirmedTerminal=$job.status -eq 'completed' -or $job.status -eq 'failed'
    if ($job.status -ne 'completed' -or -not $job.gpu_used -or -not $job.gpu_controlnet_verified -or $job.controlnet -ne $ControlNet -or $job.control_image_sha256 -ne $uploaded.sha256 -or $job.controlnet_tensors -ne 10) { throw 'Native GPU ControlNet completion/provenance was not verified.' }
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $output=Join-Path $OutputDirectory ($nativeId+'.png');$save=[IO.File]::Open($output,[IO.FileMode]::CreateNew)
    try { $save.Write($bytes,0,$bytes.Length) } finally { $save.Dispose() }
    $receipt=[ordered]@{path=$output;native_job_id=$nativeId;controlnet=$ControlNet;control_image_sha256=$job.control_image_sha256;control_strength=$ControlStrength;generation_seconds=$job.generation_seconds;windows_api_seconds=$timer.Elapsed.TotalSeconds;gpu_controlnet_verified=$job.gpu_controlnet_verified;controlnet_evals=$job.controlnet_evals;controlnet_seconds=$job.controlnet_seconds;peak_gpu_bytes=$job.peak_gpu_bytes;peak_cpu_bytes=$job.peak_cpu_bytes;sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}
    [IO.File]::WriteAllText($output+'.json',($receipt | ConvertTo-Json),[Text.UTF8Encoding]::new($false));[PSCustomObject]$receipt
} catch {
    throw ('ControlNet request failed. Do not resend automatically; check8188 /health and the known job. Uploaded map ID: '+$inputId+'. '+$_.Exception.Message)
} finally {
    if ($inputId -and $jobConfirmedTerminal) {
        try { $deleted=$client.DeleteAsync($NativeBaseUrl.TrimEnd('/')+'/api/inputs/'+$inputId).GetAwaiter().GetResult();$deleted.Dispose() } catch { }
    }
    if ($response) { $response.Dispose() };if ($content) { $content.Dispose() }
    $client.Dispose();$handler.Dispose();$secret=$null
}

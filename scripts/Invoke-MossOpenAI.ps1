# SPDX-License-Identifier: MIT
# Windows HTTP client only. MOSS Image inference runs exclusively on the PS5.
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [uri] $BaseUrl,
    [Parameter(Mandatory)] [string] $TokenFile,
    [string] $Prompt = 'A dazzling anime key visual of an adult woman with long flowing silver-white hair and violet eyes, wearing a stylish crimson jacket and a black pleated skirt, standing on a futuristic rooftop above a glowing neon city at night, a shower of shooting stars and a luminous galaxy across the sky, wind-swept hair, confident joyful smile, dramatic low-angle composition, vivid magenta and cyan rim lighting, intricate architecture, crisp clean line art, richly detailed beautiful anime illustration.',
    [string] $NegativePrompt = 'low quality, blurry',
    [ValidateSet(256,1024)] [int] $Width = 1024,
    [ValidateSet(256,1024)] [int] $Height = 1024,
    [ValidateSet(4)] [int] $Steps = 4,
    [ValidateSet(1)] [double] $CfgScale = 1,
    [ValidateSet(5)] [double] $FlowShift = 5,
    [ValidateSet(20261020)] [long] $Seed = 20261020,
    [ValidateSet('euler')] [string] $Sampler = 'euler',
    [string] $OutputDirectory = (Join-Path (Get-Location) 'output'),
    [ValidateRange(30,3600)] [int] $TimeoutSeconds = 1800,
    [switch] $ListModels
)
$ErrorActionPreference = 'Stop'
# This public verification client intentionally fixes the measured text/seed.
$canonicalPrompt = 'A dazzling anime key visual of an adult woman with long flowing silver-white hair and violet eyes, wearing a stylish crimson jacket and a black pleated skirt, standing on a futuristic rooftop above a glowing neon city at night, a shower of shooting stars and a luminous galaxy across the sky, wind-swept hair, confident joyful smile, dramatic low-angle composition, vivid magenta and cyan rim lighting, intricate architecture, crisp clean line art, richly detailed beautiful anime illustration.'
if ($Prompt -cne $canonicalPrompt -or $NegativePrompt -cne 'low quality, blurry') { throw 'Use the documented canonical verification prompt and negative prompt. Ordinary arbitrary-prompt mode is not enabled by this client.' }
. (Join-Path $PSScriptRoot 'Assert-GeneratedPng.ps1')
if ($Width -ne $Height) { throw 'The selected bounded verification profile requires a 256 or 1024 square.' }
if ($BaseUrl.Scheme -ne 'http' -or $BaseUrl.UserInfo -or $BaseUrl.Query -or $BaseUrl.Fragment) { throw 'Use an HTTP LAN base URL without credentials/query/fragment.' }
if ($BaseUrl.Port -ne 8193 -or $BaseUrl.AbsolutePath.TrimEnd('/') -cne '/v1') { throw 'Use the MOSS OpenAI gateway at http://<PS5_IP>:8193/v1.' }
$ip = $null
if (-not [Net.IPAddress]::TryParse($BaseUrl.Host,[ref]$ip) -or $ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'Use a private IPv4 LAN address.' }
$octets = $ip.GetAddressBytes()
if (-not ($octets[0] -eq 10 -or ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31) -or ($octets[0] -eq 192 -and $octets[1] -eq 168))) { throw 'Public destinations are not supported.' }
$token = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $TokenFile).Path).TrimEnd("`r","`n")
if ($token.Length -lt 32 -or $token.Length -gt 256 -or $token -cmatch '[^\x21-\x7e]') { throw 'Invalid token file.' }
Add-Type -AssemblyName System.Net.Http
$handler = [Net.Http.HttpClientHandler]::new(); $handler.AllowAutoRedirect = $false; $handler.UseProxy = $false
$client = [Net.Http.HttpClient]::new($handler); $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
$client.DefaultRequestHeaders.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer',$token)
$response = $null; $content = $null
try {
    # Fixed bounded Turbo4/CFG1 verification contract; custom sigmas are supplied by the PS5.
    $endpoint = $BaseUrl.AbsoluteUri.TrimEnd('/')
    if ($ListModels) {
        $response = $client.GetAsync("$endpoint/models").GetAwaiter().GetResult()
    } else {
        # Readiness precedes the one generation POST; this does not submit an image.
        $healthResponse = $client.GetAsync($BaseUrl.GetLeftPart([UriPartial]::Authority) + '/health').GetAwaiter().GetResult()
        try {
            if (-not $healthResponse.IsSuccessStatusCode) { throw 'Gateway health failed; no image was submitted.' }
            $health = $healthResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
            if ($health.server -cne 'ps5-ai-moss-openai-gateway' -or $health.model_id -cne 'moss-image-sft' -or $health.status -cne 'ready' -or $health.engine_available -cne $true -or $health.gpu_available -cne $true -or $health.running_jobs -ne 0 -or $health.queued_jobs -ne 0 -or $health.moss_operating_profile -ceq $true -or $health.experimental_attempt_cap_removed -ceq $true -or $health.moss_turbo_policy -cne 'euler_custom1000_910_717_24_cfg1_nsfw07_turbo1_rank144_v1' -or $health.trial_steps -ne 4 -or $health.trial_cfg_scale -ne 1 -or $health.trial_sampler -cne 'euler' -or $health.trial_scheduler -cne 'custom_sigmas_author_training_grid') { throw 'The matching idle MOSS bounded Turbo4 verification profile is required; no image was submitted.' }
        } finally { $healthResponse.Dispose() }
        if ([string]::IsNullOrWhiteSpace($Prompt) -or [Text.Encoding]::UTF8.GetByteCount($Prompt) -gt 4096 -or [Text.Encoding]::UTF8.GetByteCount($NegativePrompt) -gt 4096) { throw 'Prompt fields must fit 4096 UTF-8 bytes.' }
        $body = @{model='moss-image-sft';prompt=$Prompt;negative_prompt=$NegativePrompt;n=1;size="${Width}x${Height}";response_format='b64_json';quality='standard';steps=$Steps;cfg_scale=$CfgScale;seed=$Seed;sampler=$Sampler;flow_shift=$FlowShift} | ConvertTo-Json -Compress
        $content = [Net.Http.StringContent]::new($body,[Text.Encoding]::UTF8,'application/json')
        $watch = [Diagnostics.Stopwatch]::StartNew()
        # One POST only. A timeout is ambiguous; never automatically resubmit.
        $response = $client.PostAsync("$endpoint/images/generations",$content).GetAwaiter().GetResult()
    }
    if (-not $response.IsSuccessStatusCode) { throw "PS5 API returned HTTP $([int]$response.StatusCode); no generation retry was attempted." }
    $data = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
    if ($ListModels) { $data; return }
    $watch.Stop()
    if (@($data.data).Count -ne 1 -or -not ($data.data[0].b64_json -is [string])) { throw 'Expected one b64_json image.' }
    [byte[]] $png = [Convert]::FromBase64String($data.data[0].b64_json)
    Assert-GeneratedPng -Png $png -ExpectedWidth $Width -ExpectedHeight $Height
    $id = @($response.Headers.GetValues('x-request-id'))[0]
    $nativeId = @($response.Headers.GetValues('X-PS5-Native-Job-ID'))[0]
    if ($id -cnotmatch '^[0-9a-f]{32}$' -or $nativeId -cnotmatch '^[0-9a-f]{32}$') { throw 'Invalid job identity.' }
    $null = New-Item -ItemType Directory -Force -Path $OutputDirectory
    $path = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).Path "$id.png"
    $stream = [IO.FileStream]::new($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $stream.Write($png,0,$png.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    $result = [pscustomobject]@{model='moss-image-sft';job_id=$id;native_job_id=$nativeId;png_path=$path;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant();width=$Width;height=$Height;steps=$Steps;cfg_scale=$CfgScale;flow_shift=$FlowShift;sampler=$Sampler;seed=$Seed;client_seconds=$watch.Elapsed.TotalSeconds;inference_host=$BaseUrl.Host}
    [IO.File]::WriteAllText((Join-Path $OutputDirectory "$id.json"),($result | ConvertTo-Json) + "`n",[Text.UTF8Encoding]::new($false))
    $result
} finally {
    if ($response) { $response.Dispose() }; if ($content) { $content.Dispose() }
    $client.Dispose(); $handler.Dispose(); $token = $null
}

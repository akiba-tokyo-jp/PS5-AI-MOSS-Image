# SPDX-License-Identifier: MIT
# Windows HTTP client only. MOSS Image inference runs exclusively on the PS5.
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [uri] $BaseUrl,
    [Parameter(Mandatory)] [string] $TokenFile,
    [string] $Prompt,
    [string] $NegativePrompt = '',
    [ValidateRange(64,1024)] [int] $Width = 1024,
    [ValidateRange(64,1024)] [int] $Height = 1024,
    [ValidateSet(4)] [int] $Steps = 4,
    [ValidateSet(1)] [double] $CfgScale = 1,
    [ValidateSet(5)] [double] $FlowShift = 5,
    [ValidateRange(-1,[long]::MaxValue)] [long] $Seed = -1,
    [ValidateSet('euler')] [string] $Sampler = 'euler',
    [string] $OutputDirectory = (Join-Path (Get-Location) 'output'),
    [ValidateRange(30,3600)] [int] $TimeoutSeconds = 1800,
    [string] $ReferenceImage,
    [ValidateRange(0.01,1.0)] [double] $DenoiseStrength = 0.5,
    [switch] $ListModels
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Assert-GeneratedPng.ps1')
if (($Width % 16) -ne 0 -or ($Height % 16) -ne 0) { throw 'Image dimensions must be multiples of 16.' }
if ($BaseUrl.Scheme -ne 'http' -or $BaseUrl.UserInfo -or $BaseUrl.Query -or $BaseUrl.Fragment) { throw 'Use an HTTP LAN base URL without credentials/query/fragment.' }
if ($BaseUrl.AbsolutePath.TrimEnd('/') -cne '/v1') { throw 'Use the MOSS OpenAI gateway at http://<PS5_IP>:8193/v1.' }
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
    # Turbo adapter model recipe; prompt, seed, image shape and number of serial requests are caller controlled.
    $endpoint = $BaseUrl.AbsoluteUri.TrimEnd('/')
    if ($ListModels) {
        $response = $client.GetAsync("$endpoint/models").GetAwaiter().GetResult()
    } else {
        # Readiness precedes the one generation POST; this does not submit an image.
        $healthResponse = $client.GetAsync($BaseUrl.GetLeftPart([UriPartial]::Authority) + '/health').GetAwaiter().GetResult()
        try {
            if (-not $healthResponse.IsSuccessStatusCode) { throw 'Gateway health failed; no image was submitted.' }
            $health = $healthResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
            if ($health.server -cne 'ps5-ai-moss-openai-gateway' -or $health.model_id -cne 'moss-image-sft' -or $health.status -cne 'ready' -or $health.engine_available -cne $true -or $health.gpu_available -cne $true -or $health.running_jobs -ne 0 -or $health.queued_jobs -ne 0 -or $health.service_mode -cne $true -or $health.moss_turbo_policy -cne 'euler_custom1000_910_717_24_cfg1_nsfw07_turbo1_rank144_v1' -or $health.trial_steps -ne 4 -or $health.trial_cfg_scale -ne 1 -or $health.trial_sampler -cne 'euler' -or $health.trial_scheduler -cne 'custom_sigmas_author_training_grid') { throw 'The matching idle MOSS Turbo4 service is required; no image was submitted.' }
        } finally { $healthResponse.Dispose() }
        if ([string]::IsNullOrWhiteSpace($Prompt) -or [Text.Encoding]::UTF8.GetByteCount($Prompt) -gt 4096 -or [Text.Encoding]::UTF8.GetByteCount($NegativePrompt) -gt 4096) { throw 'Prompt fields must fit 4096 UTF-8 bytes.' }
        $fields = @{model='moss-image-sft';prompt=$Prompt;negative_prompt=$NegativePrompt;n=1;size="${Width}x${Height}";response_format='b64_json';quality='standard';steps=$Steps;cfg_scale=$CfgScale;seed=$Seed;sampler=$Sampler;flow_shift=$FlowShift}
        $targetEndpoint = "$endpoint/images/generations"
        if ($ReferenceImage) {
            if ($health.features.img2img -cne $true) { throw 'This service does not accept image input.' }
            [byte[]] $source = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $ReferenceImage).Path)
            if ($source.Length -gt 8MB -or $source.Length -lt 8) { throw 'Source PNG/JPEG must fit 8 MiB.' }
            $isPng = ($source[0] -eq 137 -and $source[1] -eq 80 -and $source[2] -eq 78 -and $source[3] -eq 71)
            $isJpeg = ($source[0] -eq 255 -and $source[1] -eq 216 -and $source[2] -eq 255)
            if (-not ($isPng -or $isJpeg)) { throw 'Source must be PNG or JPEG.' }
            $fields.reference_mode = 'img2img'; $fields.denoise_strength = $DenoiseStrength
            $content = [Net.Http.MultipartFormDataContent]::new()
            foreach ($pair in $fields.GetEnumerator()) {
                $value = [Convert]::ToString($pair.Value,[Globalization.CultureInfo]::InvariantCulture)
                $content.Add([Net.Http.StringContent]::new($value,[Text.Encoding]::UTF8),$pair.Key)
            }
            $part = [Net.Http.ByteArrayContent]::new($source)
            $part.Headers.ContentType = [Net.Http.Headers.MediaTypeHeaderValue]::new($(if ($isPng) {'image/png'} else {'image/jpeg'}))
            $content.Add($part,'image',$(if ($isPng) {'source.png'} else {'source.jpg'}))
            $targetEndpoint = "$endpoint/images/edits"
        } else {
            $content = [Net.Http.StringContent]::new(($fields | ConvertTo-Json -Compress),[Text.Encoding]::UTF8,'application/json')
        }
        $watch = [Diagnostics.Stopwatch]::StartNew()
        # One POST only. A timeout is ambiguous; never automatically resubmit.
        $response = $client.PostAsync($targetEndpoint,$content).GetAwaiter().GetResult()
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
    $resolvedSeed = [long](@($response.Headers.GetValues('X-PS5-Seed'))[0])
    if ($resolvedSeed -lt 0) { throw 'Expected the resolved native image seed.' }
    if ($id -cnotmatch '^[0-9a-f]{32}$' -or $nativeId -cnotmatch '^[0-9a-f]{32}$') { throw 'Invalid job identity.' }
    $null = New-Item -ItemType Directory -Force -Path $OutputDirectory
    $path = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).Path "$id.png"
    $stream = [IO.FileStream]::new($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $stream.Write($png,0,$png.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    $result = [pscustomobject]@{model='moss-image-sft';job_id=$id;native_job_id=$nativeId;png_path=$path;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant();width=$Width;height=$Height;steps=$Steps;cfg_scale=$CfgScale;flow_shift=$FlowShift;sampler=$Sampler;seed=$resolvedSeed;requested_seed=$Seed;mode=$(if ($ReferenceImage) {'img2img'} else {'txt2img'});client_seconds=$watch.Elapsed.TotalSeconds;inference_host=$BaseUrl.Host}
    [IO.File]::WriteAllText((Join-Path $OutputDirectory "$id.json"),($result | ConvertTo-Json) + "`n",[Text.UTF8Encoding]::new($false))
    $result
} finally {
    if ($response) { $response.Dispose() }; if ($content) { $content.Dispose() }
    $client.Dispose(); $handler.Dispose(); $token = $null
}

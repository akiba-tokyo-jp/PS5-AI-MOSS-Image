# SPDX-License-Identifier: MIT
# Windows HTTP client only. MOSS Image inference runs exclusively on the PS5.
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [uri] $BaseUrl,
    [Parameter(Mandatory)] [string] $TokenFile,
    [string] $Prompt = '1girl, adult woman, anime illustration, blue dress, gentle smile, clean lineart',
    [string] $NegativePrompt = '',
    [ValidateRange(64,1024)] [int] $Width = 1024,
    [ValidateRange(64,1024)] [int] $Height = 1024,
    [ValidateRange(1,100)] [int] $Steps = 50,
    [ValidateRange(0,30)] [double] $CfgScale = 3,
    [ValidateRange(0.1,20)] [double] $FlowShift = 5,
    [long] $Seed = -1,
    [ValidateSet('euler','euler_a','dpm++2m')] [string] $Sampler = 'euler',
    [string] $OutputDirectory = (Join-Path (Get-Location) 'output'),
    [ValidateRange(30,3600)] [int] $TimeoutSeconds = 1800,
    [switch] $ListModels
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Assert-GeneratedPng.ps1')
if ($Width % 16 -or $Height % 16) { throw 'Dimensions must be multiples of 16.' }
if ($BaseUrl.Scheme -ne 'http' -or $BaseUrl.UserInfo -or $BaseUrl.Query -or $BaseUrl.Fragment) { throw 'Use an HTTP LAN base URL without credentials/query/fragment.' }
if ($BaseUrl.Port -in @(8188,8189,8190,8191,8192)) { throw 'Use the MOSS gateway on 8193/v1; WAI/Anima/native ports are reserved.' }
$ip = $null
if (-not [Net.IPAddress]::TryParse($BaseUrl.Host,[ref]$ip) -or $ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'Use a private IPv4 LAN address.' }
$octets = $ip.GetAddressBytes()
if (-not ($octets[0] -eq 10 -or ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31) -or ($octets[0] -eq 192 -and $octets[1] -eq 168) -or $octets[0] -eq 127)) { throw 'Public destinations are not supported.' }
$token = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $TokenFile).Path).TrimEnd("`r","`n")
if ($token.Length -lt 32 -or $token.Length -gt 256 -or $token -cmatch '[^\x21-\x7e]') { throw 'Invalid token file.' }
Add-Type -AssemblyName System.Net.Http
$handler = [Net.Http.HttpClientHandler]::new(); $handler.AllowAutoRedirect = $false; $handler.UseProxy = $false
$client = [Net.Http.HttpClient]::new($handler); $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
$client.DefaultRequestHeaders.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer',$token)
$response = $null; $content = $null
try {
    $endpoint = $BaseUrl.AbsoluteUri.TrimEnd('/')
    if ($ListModels) {
        $response = $client.GetAsync("$endpoint/models").GetAwaiter().GetResult()
    } else {
        if ([Text.Encoding]::UTF8.GetByteCount($Prompt) -gt 4096 -or [Text.Encoding]::UTF8.GetByteCount($NegativePrompt) -gt 4096) { throw 'Prompt fields must fit 4096 UTF-8 bytes.' }
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
    $result = [pscustomobject]@{model='moss-image-sft';job_id=$id;native_job_id=$nativeId;png_path=$path;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant();width=$Width;height=$Height;steps=$Steps;flow_shift=$FlowShift;client_seconds=$watch.Elapsed.TotalSeconds;inference_host=$BaseUrl.Host}
    [IO.File]::WriteAllText((Join-Path $OutputDirectory "$id.json"),($result | ConvertTo-Json) + "`n",[Text.UTF8Encoding]::new($false))
    $result
} finally {
    if ($response) { $response.Dispose() }; if ($content) { $content.Dispose() }
    $client.Dispose(); $handler.Dispose(); $token = $null
}

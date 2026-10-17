# SPDX-License-Identifier: MIT
# Decode validation only. This helper performs no inference or network request.
function Assert-GeneratedPng {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [byte[]] $Png,
        [Parameter(Mandatory)] [ValidateRange(1,4096)] [int] $ExpectedWidth,
        [Parameter(Mandatory)] [ValidateRange(1,4096)] [int] $ExpectedHeight
    )
    if ($Png.LongLength -lt 45 -or $Png.LongLength -gt 64MB) {
        throw 'Downloaded PNG length is outside the allowed range.'
    }
    [byte[]] $signature = 137,80,78,71,13,10,26,10
    for ($i = 0; $i -lt 8; $i++) {
        if ($Png[$i] -ne $signature[$i]) { throw 'Invalid PNG signature.' }
    }
    [byte[]] $ihdr = 0,0,0,13,73,72,68,82
    for ($i = 0; $i -lt 8; $i++) {
        if ($Png[8 + $i] -ne $ihdr[$i]) { throw 'Invalid PNG IHDR chunk.' }
    }
    [byte[]] $iend = 0,0,0,0,73,69,78,68,174,66,96,130
    for ($i = 0; $i -lt 12; $i++) {
        if ($Png[$Png.Length - 12 + $i] -ne $iend[$i]) {
            throw 'PNG is truncated or has an invalid final IEND chunk.'
        }
    }
    $width = [uint32]$Png[16]*16777216 + [uint32]$Png[17]*65536 + [uint32]$Png[18]*256 + $Png[19]
    $height = [uint32]$Png[20]*16777216 + [uint32]$Png[21]*65536 + [uint32]$Png[22]*256 + $Png[23]
    if ($width -ne $ExpectedWidth -or $height -ne $ExpectedHeight) {
        throw 'Downloaded PNG dimensions do not match request.'
    }
    if (-not ('Ps5Ai.PngStructure' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.IO;
namespace Ps5Ai {
    public static class PngStructure {
        static readonly uint[] Table = BuildTable();
        static uint[] BuildTable() {
            var result = new uint[256];
            for (uint n = 0; n < 256; n++) {
                uint value = n;
                for (int bit = 0; bit < 8; bit++)
                    value = (value & 1) != 0 ? 0xedb88320U ^ (value >> 1) : value >> 1;
                result[n] = value;
            }
            return result;
        }
        static uint Read32(byte[] data, int at) {
            return ((uint)data[at] << 24) | ((uint)data[at+1] << 16) |
                   ((uint)data[at+2] << 8) | data[at+3];
        }
        public static void Validate(byte[] data) {
            bool pixels = false;
            for (int at = 8; at < data.Length; ) {
                if (data.Length - at < 12) throw new InvalidDataException("Truncated PNG chunk.");
                uint length = Read32(data, at);
                if (length > data.Length - at - 12)
                    throw new InvalidDataException("PNG chunk exceeds the response.");
                int size = (int)length;
                uint crc = 0xffffffffU;
                for (int n = at + 4; n < at + 8 + size; n++)
                    crc = Table[(crc ^ data[n]) & 255] ^ (crc >> 8);
                if ((crc ^ 0xffffffffU) != Read32(data, at + 8 + size))
                    throw new InvalidDataException("PNG chunk CRC mismatch.");
                string name = System.Text.Encoding.ASCII.GetString(data, at + 4, 4);
                if (at == 8 && (name != "IHDR" || size != 13))
                    throw new InvalidDataException("Invalid first PNG chunk.");
                if (at != 8 && name == "IHDR") throw new InvalidDataException("Duplicate PNG IHDR.");
                if (name == "IDAT") pixels = true;
                at += 12 + size;
                if (name == "IEND") {
                    if (size != 0 || !pixels || at != data.Length)
                        throw new InvalidDataException("Invalid PNG end or missing pixels.");
                    return;
                }
            }
            throw new InvalidDataException("Missing PNG IEND.");
        }
    }
}
'@
    }
    [Ps5Ai.PngStructure]::Validate($Png)
    Add-Type -AssemblyName System.Drawing
    $stream = [IO.MemoryStream]::new($Png, $false)
    $image = $null
    $decoded = $null
    try {
        $image = [Drawing.Image]::FromStream($stream, $false, $true)
        if ($image.RawFormat.Guid -ne [Drawing.Imaging.ImageFormat]::Png.Guid -or
            $image.Width -ne $ExpectedWidth -or $image.Height -ne $ExpectedHeight) {
            throw 'PNG decoder format or dimensions do not match request.'
        }
        $decoded = [Drawing.Bitmap]::new($image)
        $null = $decoded.GetPixel($ExpectedWidth - 1, $ExpectedHeight - 1)
    } catch {
        throw 'Downloaded PNG could not be decoded as a complete image.'
    } finally {
        if ($decoded) { $decoded.Dispose() }
        if ($image) { $image.Dispose() }
        $stream.Dispose()
    }
}

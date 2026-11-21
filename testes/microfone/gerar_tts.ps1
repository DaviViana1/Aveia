Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft Maria Desktop'); $s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile((Join-Path $PSScriptRoot 'ref-tts\referencia.wav'), $fmt)
$s.Speak((Get-Content (Join-Path $PSScriptRoot 'ref-tts\referencia.txt') -Raw -Encoding UTF8)); $s.Dispose()

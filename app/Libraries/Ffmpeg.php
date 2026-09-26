<?php

namespace App\Libraries;

class Ffmpeg
{
    public static function bin(string $name): string
    {
        foreach (['/usr/local/bin/', '/usr/bin/'] as $d) {
            if (is_executable($d . $name)) return $d . $name;
        }
        return $name;
    }

    /** @return array{duration:?float, width:?int, height:?int, vcodec:?string} */
    public static function probe(string $path): array
    {
        $r = Proc::run([self::bin('ffprobe'), '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', $path], null, 60);
        $j = json_decode($r['stdout'], true) ?: [];
        $out = ['duration' => null, 'width' => null, 'height' => null, 'vcodec' => null];
        if (isset($j['format']['duration'])) $out['duration'] = round((float) $j['format']['duration'], 3);
        foreach ($j['streams'] ?? [] as $s) {
            if (($s['codec_type'] ?? '') === 'video' && $out['vcodec'] === null) {
                $out['width']  = (int) ($s['width'] ?? 0) ?: null;
                $out['height'] = (int) ($s['height'] ?? 0) ?: null;
                $out['vcodec'] = $s['codec_name'] ?? null;
            }
        }
        return $out;
    }

    public static function hasNvenc(): bool
    {
        static $ok = null;
        if ($ok === null) {
            $r  = Proc::run([self::bin('ffmpeg'), '-hide_banner', '-encoders'], null, 30);
            $ok = str_contains($r['stdout'], 'h264_nvenc') && (file_exists('/dev/dxg') || file_exists('/dev/nvidia0'));
        }
        return $ok;
    }

    /**
     * Runs ffmpeg with -progress on stderr and reports 0..100 via $onProgress using the known duration.
     * @param callable(int):void|null $onProgress
     */
    public static function run(array $args, ?float $duration, ?callable $onProgress): array
    {
        $cmd = array_merge([self::bin('ffmpeg'), '-y', '-hide_banner', '-loglevel', 'error', '-nostats', '-progress', 'pipe:2'], $args);
        $last = -1;
        return Proc::run($cmd, static function (string $line) use ($duration, $onProgress, &$last): void {
            if (! $onProgress || ! $duration || ! str_starts_with($line, 'out_time_ms=')) return;
            $pct = (int) min(100, max(0, ((int) substr($line, 12)) / 1_000_000 / $duration * 100));
            if ($pct !== $last) { $last = $pct; $onProgress($pct); }
        });
    }

    /** 16 kHz mono WAV, what WhisperX expects. */
    public static function extractAudio(string $src, string $out, ?float $duration, ?callable $onProgress): void
    {
        $r = self::run(['-i', $src, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', $out], $duration, $onProgress);
        if ($r['code'] !== 0 || ! is_file($out)) {
            throw new \RuntimeException('audio extract failed: ' . mb_substr($r['stderr'], -800));
        }
    }

    /** Clean H.264 High / AAC stereo mp4 (max 1080p, faststart, first video + audio stream only). */
    public static function makeProxy(string $src, string $out, ?float $duration, ?callable $onProgress): void
    {
        $enc = self::hasNvenc()
            ? ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '26', '-maxrate', '8M', '-bufsize', '16M', '-profile:v', 'high']
            : ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-profile:v', 'high'];
        $args = array_merge(['-i', $src, '-map', '0:v:0', '-map', '0:a:0?', '-sn', '-dn',
            '-vf', "scale=-2:'min(ih,1080)'", '-pix_fmt', 'yuv420p', '-g', '60'], $enc,
            ['-c:a', 'aac', '-b:a', '160k', '-ac', '2', '-movflags', '+faststart', $out]);
        $r = self::run($args, $duration, $onProgress);
        if ($r['code'] !== 0 || ! is_file($out)) {
            throw new \RuntimeException('proxy encode failed: ' . mb_substr($r['stderr'], -800));
        }
    }
}

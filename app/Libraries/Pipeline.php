<?php

namespace App\Libraries;

use App\Models\JobModel;
use App\Models\VideoModel;
use Config\MovTrans;

/**
 * Runs one job: prepare (audio + optional proxy) -> transcribe (WhisperX)
 * -> scene (Qwen3-VL frame analysis) -> translate (Qwen3-VL Korean subtitles).
 * Each stage reports its own 0..100 into jobs.progress; jobs.stage names the running stage.
 */
class Pipeline
{
    public const STAGES = ['prepare', 'transcribe', 'scene', 'translate'];

    private JobModel $jobs;
    private VideoModel $videos;
    private MovTrans $cfg;
    private array $logLines = [];

    public function __construct()
    {
        $this->jobs   = new JobModel();
        $this->videos = new VideoModel();
        $this->cfg    = config('MovTrans');
    }

    /** Atomically claim the oldest queued job (MySQL LAST_INSERT_ID trick). */
    public function claim(): ?array
    {
        $db = $this->jobs->db;
        $db->query("UPDATE jobs SET id = LAST_INSERT_ID(id), status = 'running', started_at = NOW(), updated_at = NOW() WHERE status = 'queued' ORDER BY id ASC LIMIT 1");
        if ($db->affectedRows() < 1) return null;
        $id = (int) $db->query('SELECT LAST_INSERT_ID() AS id')->getRow()->id;
        return $this->jobs->find($id);
    }

    /** Single worker: any 'running' row at startup was cut off by a restart (deploy), so queue it again. */
    public function requeueStale(): int
    {
        $db = $this->jobs->db;
        $db->query("UPDATE jobs SET status = 'queued', stage = 'queued', progress = 0, updated_at = NOW() WHERE status = 'running'");
        return $db->affectedRows();
    }

    public function run(array $job, ?callable $echo = null): void
    {
        $echo ??= static function (string $m): void {};
        $this->logLines = [];
        $jobId = (int) $job['id'];
        $video = $this->videos->find((int) $job['video_id']);
        try {
            if (! $video) throw new \RuntimeException('video row missing');
            $this->videos->update($video['id'], ['status' => 'processing']);
            $dir = VideoModel::dir((int) $video['id']);
            $src = $dir . '/' . $video['filename'];
            if (! is_file($src)) throw new \RuntimeException('source file missing: ' . $src);
            $dur = $video['duration'] !== null ? (float) $video['duration'] : null;

            $this->stage($jobId, 'prepare');
            $this->prepare($jobId, $video, $dir, $src, $dur, $echo);

            $this->stage($jobId, 'transcribe');
            $this->python($jobId, $echo, [
                ROOTPATH . 'bin/transcribe.py', '--audio', $dir . '/audio.wav', '--lang', $video['lang'],
                '--model', $this->cfg->whisperModel, '--compute', $this->cfg->whisperCompute, '--out-dir', $dir,
                '--hf-token', $this->cfg->hfToken,
                '--fill-model', $video['lang'] === 'ja' ? $this->cfg->whisperFillModel : '',
            ]);
            $this->requireFile($dir . '/segments.json');

            if ($video['lang'] === 'ko') {
                // already Korean: the transcript is the final subtitle
                copy($dir . '/orig.srt', $dir . '/ko.srt');
                copy($dir . '/orig.vtt', $dir . '/ko.vtt');
            } else {
                $this->stage($jobId, 'scene');
                $this->python($jobId, $echo, [
                    ROOTPATH . 'bin/scene.py', '--video', $dir . '/proxy.mp4', '--segments', $dir . '/segments.json',
                    '--out', $dir . '/scenes.json', '--frames-dir', $dir . '/frames',
                    '--ollama', $this->cfg->ollamaUrl, '--model', $this->cfg->vlModel, '--max-frames', (string) $this->cfg->maxFrames,
                ]);
                $this->requireFile($dir . '/scenes.json');

                $this->stage($jobId, 'translate');
                $this->python($jobId, $echo, [
                    ROOTPATH . 'bin/translate.py', '--segments', $dir . '/segments.json', '--scenes', $dir . '/scenes.json',
                    '--lang', $video['lang'], '--out-dir', $dir, '--ollama', $this->cfg->ollamaUrl, '--model', $this->cfg->vlModel,
                    '--rating', $video['rating'] ?? 'rated',
                ]);
            }
            $this->requireFile($dir . '/ko.vtt');
            foreach (glob($dir . '/*.{srt,vtt,json}', GLOB_BRACE) ?: [] as $f) Storage::relax($f);

            $this->jobs->update($jobId, ['status' => 'done', 'stage' => 'done', 'progress' => 100, 'finished_at' => date('Y-m-d H:i:s'), 'log' => $this->log()]);
            $this->videos->update($video['id'], ['status' => 'done']);
            $echo("job {$jobId} done");
        } catch (\Throwable $e) {
            $this->jobs->update($jobId, ['status' => 'failed', 'error' => mb_substr($e->getMessage(), 0, 2000), 'finished_at' => date('Y-m-d H:i:s'), 'log' => $this->log()]);
            if ($video) $this->videos->update($video['id'], ['status' => 'failed']);
            $echo("job {$jobId} FAILED: " . $e->getMessage());
        }
    }

    private function prepare(int $jobId, array $video, string $dir, string $src, ?float $dur, callable $echo): void
    {
        // every upload gets a clean H.264/AAC mp4 (playback + frame grabs); the source is kept untouched
        $split = 15;
        if (! is_file($dir . '/audio.wav')) {
            Ffmpeg::extractAudio($src, $dir . '/audio.wav', $dur, fn (int $p) => $this->progress($jobId, (int) ($p * $split / 100)));
        }
        Storage::relax($dir . '/audio.wav');
        if (! $video['has_proxy'] || ! is_file($dir . '/proxy.mp4')) {
            $echo('encoding H.264 proxy (' . ($video['vcodec'] ?? '?') . ')');
            Ffmpeg::makeProxy($src, $dir . '/proxy.mp4', $dur, fn (int $p) => $this->progress($jobId, $split + (int) ($p * (100 - $split) / 100)));
            Storage::relax($dir . '/proxy.mp4');
            $this->videos->update($video['id'], ['has_proxy' => 1]);
        }
        $this->progress($jobId, 100);
    }

    private function python(int $jobId, callable $echo, array $scriptArgs): void
    {
        $env = [];
        if ($this->cfg->pylibs !== '' && is_dir($this->cfg->pylibs)) $env['PYTHONPATH'] = $this->cfg->pylibs;
        $env['PYTHONUNBUFFERED'] = '1';
        $r = Proc::run(array_merge([$this->cfg->python], $scriptArgs), function (string $line) use ($jobId, $echo): void {
            if (preg_match('/^progress (\d{1,3})$/', $line, $m)) {
                $this->progress($jobId, (int) $m[1]);
                return;
            }
            $this->logLines[] = $line;
            if (count($this->logLines) > 200) array_shift($this->logLines);
            $this->jobs->update($jobId, ['log' => $this->log()]);
            $echo('  ' . $line);
        }, 0, $env);
        if ($r['code'] !== 0) {
            throw new \RuntimeException(basename($scriptArgs[0]) . ' exited ' . $r['code'] . ': ' . mb_substr(trim($r['stderr']), -600));
        }
    }

    private function stage(int $jobId, string $stage): void
    {
        $this->jobs->update($jobId, ['stage' => $stage, 'progress' => 0]);
    }

    private function progress(int $jobId, int $pct): void
    {
        $this->jobs->update($jobId, ['progress' => max(0, min(100, $pct))]);
    }

    private function requireFile(string $path): void
    {
        if (! is_file($path)) throw new \RuntimeException('expected output missing: ' . basename($path));
    }

    private function log(): string
    {
        return mb_substr(implode("\n", $this->logLines), -60000);
    }
}

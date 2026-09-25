<?php

namespace App\Libraries;

/** Runs a child process, streaming stderr lines to a callback (used for "progress N" reporting). */
class Proc
{
    /**
     * @param callable(string):void|null $onStderrLine
     * @return array{code:int, stdout:string, stderr:string}
     */
    public static function run(array $args, ?callable $onStderrLine = null, int $timeout = 0, array $env = []): array
    {
        $spec = [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => ['pipe', 'w']];
        $env  = array_merge(['PATH' => getenv('PATH') ?: '/usr/local/bin:/usr/bin:/bin', 'HOME' => getenv('HOME') ?: '/tmp'], $env);
        $p    = proc_open($args, $spec, $pipes, null, $env);
        if (! is_resource($p)) {
            return ['code' => -1, 'stdout' => '', 'stderr' => 'proc_open failed'];
        }
        fclose($pipes[0]);
        stream_set_blocking($pipes[1], false);
        stream_set_blocking($pipes[2], false);
        $stdout = ''; $stderr = ''; $buf = ''; $start = time();
        $pump = static function () use (&$pipes, &$stdout, &$stderr, &$buf, $onStderrLine): void {
            $stdout .= (string) stream_get_contents($pipes[1]);
            $chunk = (string) stream_get_contents($pipes[2]);
            if ($chunk === '') return;
            $stderr .= $chunk;
            $buf .= $chunk;
            while (($nl = strpos($buf, "\n")) !== false) {
                $line = rtrim(substr($buf, 0, $nl), "\r");
                $buf  = substr($buf, $nl + 1);
                if ($onStderrLine && $line !== '') $onStderrLine($line);
            }
        };
        while (true) {
            $pump();
            $st = proc_get_status($p);
            if (! $st['running']) break;
            if ($timeout > 0 && time() - $start > $timeout) { proc_terminate($p, 9); $stderr .= "\ntimeout"; break; }
            usleep(100000);
        }
        $pump();
        if ($buf !== '' && $onStderrLine) $onStderrLine(rtrim($buf, "\r"));
        fclose($pipes[1]); fclose($pipes[2]);
        $code = proc_close($p);
        return ['code' => $st['running'] ? -1 : ($st['exitcode'] ?? $code), 'stdout' => $stdout, 'stderr' => $stderr];
    }
}

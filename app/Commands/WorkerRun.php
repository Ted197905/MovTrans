<?php

namespace App\Commands;

use App\Libraries\Pipeline;
use CodeIgniter\CLI\BaseCommand;
use CodeIgniter\CLI\CLI;

class WorkerRun extends BaseCommand
{
    protected $group       = 'movtrans';
    protected $name        = 'worker:run';
    protected $description = 'Process queued subtitle jobs. --once handles a single job and exits.';
    protected $options     = ['--once' => 'Process at most one job then exit', '--sleep' => 'Idle poll interval seconds (default 2)'];

    public function run(array $params)
    {
        $once  = array_key_exists('once', $params);
        $sleep = (int) ($params['sleep'] ?? 2) ?: 2;
        $pipe  = new Pipeline();
        $log   = static fn (string $m) => CLI::write('[' . date('H:i:s') . '] ' . $m);
        $log('worker started (pid ' . getmypid() . ')');
        if ($n = $pipe->requeueStale()) $log("requeued {$n} interrupted job(s)");
        while (true) {
            $job = $pipe->claim();
            if ($job) {
                $log("job {$job['id']} video {$job['video_id']}");
                $pipe->run($job, $log);
                if ($once) return;
                if (is_file(WRITEPATH . 'restart.flag')) {  // deploy.sh asked for a restart while we were busy
                    @unlink(WRITEPATH . 'restart.flag');
                    $log('restarting after deploy');
                    return;
                }
                continue;
            }
            if ($once) { $log('no queued jobs'); return; }
            sleep($sleep);
        }
    }
}

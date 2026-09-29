<?php

namespace App\Models;

use CodeIgniter\Model;

class JobModel extends Model
{
    protected $table         = 'jobs';
    protected $allowedFields = ['video_id', 'status', 'stage', 'progress', 'log', 'error', 'warnings', 'started_at', 'finished_at'];
    protected $useTimestamps = true;

    public function latestFor(int $videoId): ?array
    {
        return $this->where('video_id', $videoId)->orderBy('id', 'DESC')->first();
    }
}

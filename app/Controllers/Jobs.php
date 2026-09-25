<?php

namespace App\Controllers;

use App\Models\JobModel;
use App\Models\VideoModel;

class Jobs extends BaseController
{
    /** GET /api/jobs/{videoId} -> latest job for the video (polled by the progress page) */
    public function show(int $videoId)
    {
        $job = (new JobModel())->latestFor($videoId);
        if (! $job) {
            return $this->response->setStatusCode(404)->setJSON(['error' => 'not found']);
        }
        $video = (new VideoModel())->find($videoId);
        return $this->response->setJSON([
            'job'       => $job,
            'status'    => $video['status'] ?? null,
            'subtitles' => VideoModel::subtitles($videoId),
        ]);
    }
}

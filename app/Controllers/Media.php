<?php

namespace App\Controllers;

use App\Models\VideoModel;
use CodeIgniter\Exceptions\PageNotFoundException;

/** Files live outside the web root; nginx streams them after this auth check (X-Accel-Redirect). */
class Media extends BaseController
{
    private function accel(int $id, string $file, string $contentType, ?string $downloadName = null)
    {
        if (! is_file(VideoModel::dir($id) . '/' . $file)) {
            throw PageNotFoundException::forPageNotFound();
        }
        $resp = $this->response
            ->setHeader('X-Accel-Redirect', '/internal-media/' . $id . '/' . $file)
            ->setHeader('Content-Type', $contentType)
            ->setHeader('Cache-Control', 'private, no-cache');
        if ($downloadName !== null) {
            $resp->setHeader('Content-Disposition', 'attachment; filename*=UTF-8\'\'' . rawurlencode($downloadName));
        }
        return $resp;
    }

    /** Browser-playable file: the H.264 proxy when one was made, else the source. */
    public function video(int $id)
    {
        $video = (new VideoModel())->find($id) ?? throw PageNotFoundException::forPageNotFound();
        if ($video['has_proxy']) {
            return $this->accel($id, 'proxy.mp4', 'video/mp4');
        }
        $ext  = pathinfo($video['filename'], PATHINFO_EXTENSION);
        $mime = ['webm' => 'video/webm', 'mkv' => 'video/x-matroska', 'mov' => 'video/quicktime'][$ext] ?? 'video/mp4';
        return $this->accel($id, $video['filename'], $mime);
    }

    /** /media/{id}/sub/{orig|ko}.{vtt|srt}  (?dl to download) */
    public function subtitle(int $id, string $track, string $ext)
    {
        if (! in_array($track, ['orig', 'ko'], true) || ! in_array($ext, ['vtt', 'srt'], true)) {
            throw PageNotFoundException::forPageNotFound();
        }
        $video = (new VideoModel())->find($id) ?? throw PageNotFoundException::forPageNotFound();
        $name  = $this->request->getGet('dl') !== null ? $video['title'] . '.' . $track . '.' . $ext : null;
        return $this->accel($id, "{$track}.{$ext}", $ext === 'vtt' ? 'text/vtt; charset=utf-8' : 'text/plain; charset=utf-8', $name);
    }
}

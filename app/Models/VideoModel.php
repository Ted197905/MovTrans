<?php

namespace App\Models;

use CodeIgniter\Model;

class VideoModel extends Model
{
    protected $table         = 'videos';
    protected $allowedFields = ['title', 'filename', 'size', 'duration', 'width', 'height', 'vcodec', 'has_proxy', 'lang', 'rating', 'status'];
    protected $useTimestamps = true;

    public static function dir(int $id): string
    {
        return WRITEPATH . 'media/' . $id;
    }

    /** Subtitle files present for a video: [['orig','srt'], ...] */
    public static function subtitles(int $id): array
    {
        $out = [];
        foreach (['orig', 'ko'] as $track) {
            foreach (['vtt', 'srt'] as $ext) {
                if (is_file(self::dir($id) . "/{$track}.{$ext}")) $out[] = [$track, $ext];
            }
        }
        return $out;
    }
}

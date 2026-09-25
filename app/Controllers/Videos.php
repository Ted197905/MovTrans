<?php

namespace App\Controllers;

use App\Libraries\Storage;
use App\Models\JobModel;
use App\Models\VideoModel;
use CodeIgniter\Exceptions\PageNotFoundException;

class Videos extends BaseController
{
    public function index()
    {
        $videos = new VideoModel();
        $jobs   = new JobModel();
        $rows   = $videos->orderBy('id', 'DESC')->findAll();
        foreach ($rows as &$r) {
            $r['job'] = $jobs->latestFor((int) $r['id']);
        }
        return view('videos/index', ['title' => '영상', 'videos' => $rows, 'langs' => self::LANGS]);
    }

    public function show(int $id)
    {
        $video = (new VideoModel())->find($id) ?? throw PageNotFoundException::forPageNotFound();
        return view('videos/show', [
            'title'     => $video['title'],
            'video'     => $video,
            'job'       => (new JobModel())->latestFor($id),
            'subtitles' => VideoModel::subtitles($id),
            'langs'     => self::LANGS,
        ]);
    }

    public function delete(int $id)
    {
        $videos = new VideoModel();
        if ($videos->find($id)) {
            (new JobModel())->where('video_id', $id)->delete();
            $videos->delete($id);
            Storage::removeDir(VideoModel::dir($id));
        }
        return redirect()->to('/videos');
    }

    /** Queue a fresh pipeline run (keeps the source file, drops generated outputs). */
    public function rerun(int $id)
    {
        $videos = new VideoModel();
        $video  = $videos->find($id) ?? throw PageNotFoundException::forPageNotFound();
        $lang   = (string) $this->request->getPost('lang');
        if (isset(self::LANGS[$lang])) {
            $videos->update($id, ['lang' => $lang]);
        }
        $dir = VideoModel::dir($id);
        foreach (['audio.wav', 'segments.json', 'scenes.json', 'orig.srt', 'orig.vtt', 'ko.srt', 'ko.vtt'] as $f) {
            @unlink($dir . '/' . $f);
        }
        Storage::removeDir($dir . '/frames');
        $videos->update($id, ['status' => 'queued']);
        (new JobModel())->insert(['video_id' => $id]);
        return redirect()->to('/videos/' . $id);
    }

    /** Source languages offered in the UI (WhisperX language codes). */
    public const LANGS = [
        'en' => 'English', 'ja' => '日本語', 'zh' => '中文', 'es' => 'Español', 'fr' => 'Français',
        'de' => 'Deutsch', 'ru' => 'Русский', 'pt' => 'Português', 'it' => 'Italiano', 'th' => 'ไทย',
        'vi' => 'Tiếng Việt', 'id' => 'Bahasa Indonesia', 'ko' => '한국어',
    ];
}

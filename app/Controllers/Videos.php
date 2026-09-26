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
        $queued = array_column($jobs->select('id')->where('status', 'queued')->orderBy('id', 'ASC')->findAll(), 'id');
        foreach ($rows as &$r) {
            $r['job'] = $jobs->latestFor((int) $r['id']);
            $pos = $r['job'] && $r['job']['status'] === 'queued' ? array_search($r['job']['id'], $queued) : false;
            $r['queuePos'] = $pos === false ? null : $pos + 1;
        }
        return view('videos/index', ['title' => '영상', 'videos' => $rows, 'langs' => self::LANGS, 'ratings' => self::RATINGS]);
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
            'ratings'   => self::RATINGS,
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

    /** POST /videos/{id}/start: add to the job queue with the stored language/rating. */
    public function start(int $id)
    {
        $video = (new VideoModel())->find($id) ?? throw PageNotFoundException::forPageNotFound();
        $this->enqueue($video);
        return redirect()->back();
    }

    /** Queue a fresh pipeline run, optionally changing language/rating. */
    public function rerun(int $id)
    {
        $videos = new VideoModel();
        $video  = $videos->find($id) ?? throw PageNotFoundException::forPageNotFound();
        if (in_array($video['status'], ['queued', 'processing'], true)) {
            return redirect()->to('/videos/' . $id);
        }
        $lang   = (string) $this->request->getPost('lang');
        if (isset(self::LANGS[$lang])) {
            $videos->update($id, ['lang' => $lang]);
        }
        $rating = (string) $this->request->getPost('rating');
        if (isset(self::RATINGS[$rating])) {
            $videos->update($id, ['rating' => $rating]);
        }
        $this->enqueue($video);
        return redirect()->to('/videos/' . $id);
    }

    /** Drop generated outputs (source is kept) and add a queued job. No-op while queued/running. */
    private function enqueue(array $video): void
    {
        if (in_array($video['status'], ['queued', 'processing'], true)) return;
        $id  = (int) $video['id'];
        $dir = VideoModel::dir($id);
        foreach (['audio.wav', 'segments.json', 'scenes.json', 'orig.srt', 'orig.vtt', 'ko.srt', 'ko.vtt'] as $f) {
            @unlink($dir . '/' . $f);
        }
        Storage::removeDir($dir . '/frames');
        (new VideoModel())->update($id, ['status' => 'queued']);
        (new JobModel())->insert(['video_id' => $id]);
    }

    /** Translation explicitness (passed to bin/translate.py --rating). */
    public const RATINGS = ['rated' => 'RATED (미국 개봉영화 수준)', 'unrated' => 'UNRATED (성인 비디오 수준)'];

    /** Source languages offered in the UI (WhisperX language codes). */
    public const LANGS = [
        'en' => 'English', 'ja' => '日本語', 'zh' => '中文', 'es' => 'Español', 'fr' => 'Français',
        'de' => 'Deutsch', 'ru' => 'Русский', 'pt' => 'Português', 'it' => 'Italiano', 'th' => 'ไทย',
        'vi' => 'Tiếng Việt', 'id' => 'Bahasa Indonesia', 'ko' => '한국어',
    ];
}

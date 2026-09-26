<?php

namespace App\Controllers;

use App\Libraries\Ffmpeg;
use App\Libraries\Storage;
use App\Models\JobModel;
use App\Models\VideoModel;

/**
 * Chunked upload: the browser slices the file into fixed parts so one request never
 * exceeds CHUNK_MAX and nginx/php limits stay small regardless of the video size.
 */
class Upload extends BaseController
{
    private const CHUNK_MAX = 16 * 1024 * 1024;

    private function tmpDir(string $uploadId): string
    {
        return WRITEPATH . 'chunks/' . $uploadId;
    }

    private static function validId(string $id): bool
    {
        return (bool) preg_match('/^[a-f0-9]{32}$/', $id);
    }

    /** POST /api/upload/init {name, size, lang, rating} */
    public function init()
    {
        $cfg  = config('MovTrans');
        $in   = $this->request->getJSON(true) ?: [];
        $name = (string) ($in['name'] ?? '');
        $size = (int) ($in['size'] ?? 0);
        $lang = (string) ($in['lang'] ?? '');
        $rating = (string) ($in['rating'] ?? 'rated');
        $ext  = strtolower(pathinfo($name, PATHINFO_EXTENSION));
        if (! in_array($ext, $cfg->allowedExt, true)) {
            return $this->response->setStatusCode(415)->setJSON(['error' => '지원하지 않는 파일 형식입니다: .' . $ext]);
        }
        if ($size <= 0 || $size > $cfg->maxUploadBytes) {
            return $this->response->setStatusCode(413)->setJSON(['error' => '파일 크기가 허용 범위를 벗어났습니다.']);
        }
        if (! isset(Videos::LANGS[$lang])) {
            return $this->response->setStatusCode(400)->setJSON(['error' => '원어를 선택하세요.']);
        }
        if (! isset(Videos::RATINGS[$rating])) {
            return $this->response->setStatusCode(400)->setJSON(['error' => '번역 수위를 선택하세요.']);
        }
        $uploadId = bin2hex(random_bytes(16));
        $dir = $this->tmpDir($uploadId);
        if (! mkdir($dir, 0775, true)) {
            return $this->response->setStatusCode(500)->setJSON(['error' => '임시 디렉토리를 만들 수 없습니다.']);
        }
        Storage::relax(dirname($dir));
        file_put_contents($dir . '/meta.json', json_encode(['name' => $name, 'size' => $size, 'ext' => $ext, 'lang' => $lang, 'rating' => $rating]));
        return $this->response->setJSON(['ok' => true, 'uploadId' => $uploadId, 'chunkSize' => 8 * 1024 * 1024]);
    }

    /** POST /api/upload/chunk  multipart: uploadId, index, chunk */
    public function chunk()
    {
        $uploadId = (string) $this->request->getPost('uploadId');
        $index    = (int) $this->request->getPost('index');
        if (! self::validId($uploadId) || $index < 0 || $index > 100000) {
            return $this->response->setStatusCode(400)->setJSON(['error' => 'bad request']);
        }
        $dir = $this->tmpDir($uploadId);
        if (! is_dir($dir)) {
            return $this->response->setStatusCode(404)->setJSON(['error' => '업로드 세션이 없습니다. 다시 시도하세요.']);
        }
        $file = $this->request->getFile('chunk');
        if (! $file || ! $file->isValid()) {
            return $this->response->setStatusCode(400)->setJSON(['error' => $file ? $file->getErrorString() : '조각이 없습니다.']);
        }
        if ($file->getSize() > self::CHUNK_MAX) {
            return $this->response->setStatusCode(413)->setJSON(['error' => '조각이 너무 큽니다.']);
        }
        $file->move($dir, sprintf('%06d.part', $index), true);
        return $this->response->setJSON(['ok' => true, 'index' => $index]);
    }

    /** POST /api/upload/finish {uploadId, total} -> creates the video row and queues a job */
    public function finish()
    {
        $in       = $this->request->getJSON(true) ?: [];
        $uploadId = (string) ($in['uploadId'] ?? '');
        $total    = (int) ($in['total'] ?? 0);
        if (! self::validId($uploadId) || $total < 1) {
            return $this->response->setStatusCode(400)->setJSON(['error' => 'bad request']);
        }
        $dir  = $this->tmpDir($uploadId);
        $meta = @json_decode((string) @file_get_contents($dir . '/meta.json'), true);
        if (! is_dir($dir) || ! is_array($meta)) {
            return $this->response->setStatusCode(404)->setJSON(['error' => '업로드 세션이 없습니다.']);
        }
        for ($i = 0; $i < $total; $i++) {
            if (! is_file($dir . '/' . sprintf('%06d.part', $i))) {
                Storage::removeDir($dir);
                return $this->response->setStatusCode(422)->setJSON(['error' => '누락된 조각이 있습니다 (' . $i . ').']);
            }
        }
        $videos = new VideoModel();
        $id = $videos->insert([
            'title'    => mb_substr(pathinfo($meta['name'], PATHINFO_FILENAME), 0, 255),
            'filename' => 'source.' . $meta['ext'],
            'size'     => (int) $meta['size'],
            'lang'     => $meta['lang'],
            'rating'   => $meta['rating'] ?? 'rated',
        ]);
        $mdir = VideoModel::dir($id);
        if (! is_dir($mdir) && ! mkdir($mdir, 0775, true)) {
            $videos->delete($id);
            return $this->response->setStatusCode(500)->setJSON(['error' => '저장 디렉토리를 만들 수 없습니다.']);
        }
        Storage::relax(dirname($mdir));
        Storage::relax($mdir);
        $target = $mdir . '/source.' . $meta['ext'];
        $out = fopen($target, 'wb');
        for ($i = 0; $i < $total; $i++) {
            $p = $dir . '/' . sprintf('%06d.part', $i);
            $in_ = fopen($p, 'rb');
            stream_copy_to_stream($in_, $out);
            fclose($in_);
        }
        fclose($out);
        Storage::relax($target);
        Storage::removeDir($dir);

        $info = Ffmpeg::probe($target);
        $videos->update($id, [
            'duration' => $info['duration'], 'width' => $info['width'], 'height' => $info['height'], 'vcodec' => $info['vcodec'],
        ]);
        (new JobModel())->insert(['video_id' => $id]);
        return $this->response->setJSON(['ok' => true, 'id' => $id, 'url' => site_url('videos/' . $id)]);
    }

    /** POST /api/upload/abort {uploadId} */
    public function abort()
    {
        $in = $this->request->getJSON(true) ?: [];
        $id = (string) ($in['uploadId'] ?? '');
        if (self::validId($id)) Storage::removeDir($this->tmpDir($id));
        return $this->response->setJSON(['ok' => true]);
    }
}

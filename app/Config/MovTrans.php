<?php

namespace Config;

use CodeIgniter\Config\BaseConfig;

/** Pipeline settings. Override any of these in .env as movtrans.<name>. */
class MovTrans extends BaseConfig
{
    /** Python interpreter for bin/*.py. */
    public string $python = '/usr/bin/python3';

    /** Extra sys.path entry (pip install --target ./pylibs). Empty to skip. */
    public string $pylibs = ROOTPATH . 'pylibs';

    /** WhisperX model and compute type. */
    public string $whisperModel = 'large-v3';
    public string $whisperCompute = 'float16';

    /** Ollama endpoint and the vision model used for scene analysis + translation. */
    public string $ollamaUrl = 'http://127.0.0.1:11434';
    public string $vlModel   = 'huihui_ai/qwen3-vl-abliterated:8b-instruct';

    /** Max frames sent to the VL model per video (segments are sampled evenly beyond this). */
    public int $maxFrames = 300;

    /** Upload limits. */
    public int $maxUploadBytes = 10 * 1024 * 1024 * 1024;
    public array $allowedExt = ['mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v', 'ts', 'wmv', 'flv'];
}

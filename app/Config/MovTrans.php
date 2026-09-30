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

    /** Hugging Face token for pyannote speaker diarization (gated models; empty: Sortformer or pitch only). */
    public string $hfToken = '';
    /** auto = NVIDIA Sortformer (pyenv/nemo) if installed, else pyannote (token), else pitch; both = Sortformer + pyannote cross-check (logged). */
    public string $diarizer = 'auto';   // 'both' adds the pyannote cross-check (~5 min per 4 h, log only)

    /** Second translation pass (감수) over the draft subtitles: ~25 min per 4 h video on a 3080, off = draft only. */
    public bool $polish = true;

    /** Whisper model and compute type. */
    public string $whisperModel = 'large-v3';
    /** Second model for spans the first left empty (empty string disables). Japanese: kotoba-whisper. */
    public string $whisperFillModel = 'kotoba-tech/kotoba-whisper-v2.0-faster';
    public string $whisperCompute = 'float16';

    /** Ollama endpoint and the vision model used for scene analysis + translation. */
    public string $ollamaUrl = 'http://127.0.0.1:11434';
    public string $vlModel   = 'huihui_ai/qwen3-vl-abliterated:8b-instruct';
    /** Text model for the style guide and the translation itself (a 35B-A3B MoE runs split CPU/GPU at ~40 s per 25 lines). */
    public string $textModel = 'huihui_ai/Qwen3.6-abliterated:35b-a3b-q4_K';

    /** Max frames sent to the VL model per video (segments are sampled evenly beyond this). */
    public int $maxFrames = 300;

    /** Upload limits. */
    public int $maxUploadBytes = 10 * 1024 * 1024 * 1024;
    public array $allowedExt = ['mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v', 'ts', 'wmv', 'flv'];
}

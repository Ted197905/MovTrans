<?php

if (! function_exists('asset_url')) {
    /**
     * Asset URL with the file's modification time appended, so a deploy invalidates
     * the browser cache instead of leaving people on a stale script.
     */
    function asset_url(string $path): string
    {
        $file = rtrim(FCPATH, '/') . '/' . ltrim($path, '/');
        $v    = is_file($file) ? filemtime($file) : null;
        return base_url($path) . ($v ? '?v=' . $v : '');
    }
}

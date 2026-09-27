# MovTrans

영상을 업로드하면 원어 자막(faster-whisper)과 장면 이해 기반 한국어 자막(Qwen3-VL + Qwen3.6)을 만들어 주는 웹 서비스.
단일 관리자 계정, CodeIgniter 4 + MySQL, 무거운 처리는 systemd 워커가 담당한다.

## 흐름

1. 업로드 (청크 8 MB, 최대 10 GB) + 원어/번역 수위 선택 -> `videos` 행 (status `uploaded`). 목록의 "진행" 으로 `jobs` 행 생성
2. 워커 `php spark worker:run` 이 `jobs` 를 2초마다 폴링해 단계별로 실행
   - `prepare`    ffmpeg 16 kHz 오디오 추출 + 모든 영상을 H.264/AAC 프록시(최대 1080p)로 재인코딩
   - `transcribe` `bin/transcribe.py` faster-whisper large-v3 를 발화 시작점 기준 30초 클립으로 실행, 환각/신음 필터,
                  걸러진 창 재시도, 빈 구간은 kotoba-whisper 로 보충(+wav2vec2 정렬), 화자 분리(NVIDIA Sortformer 주,
                  pyannote 교차검증, 둘 다 없으면 피치 F/M) -> 태그 S1..Sn + 음높이 힌트, 단어 타임스탬프로 자막 큐 재구성,
                  큐마다 wav2vec2 강제 정렬로 시작/끝 보정 (`bin/timing.py`) -> `segments.json`(cast, sounds 포함), `orig.srt/vtt`
   - `scene`      `bin/scene.py` 큐마다 프레임 1장을 Qwen3-VL(Ollama)로 분석 + 영상 개요 -> `scenes.json`
   - `screen`     `bin/ocr.py` 1fps EasyOCR 검출 -> Qwen3-VL 판독: 화면 속 자막/캡션만 -> `screen.json` (실패해도 잡은 계속)
   - `translate`  `bin/translate.py` 스타일 가이드(등장인물/말투) 생성 -> 25줄 배치 번역(장면 노트, 앞뒤 문맥) -> 재번역
                  -> `ko.srt/vtt`, SDH 트랙 `ko.sdh.srt/vtt`(화자 표시, [신음]/[웃음]), `style.txt`
3. 진행 페이지가 `/api/jobs/{id}` 를 폴링해 단계별 진행 바 표시, 완료되면 `<video>` + `<track>` 로 재생 (한국어 / SDH / 원어)

파일은 `writable/media/{id}/` 에 두고 nginx `X-Accel-Redirect` 로 전송한다.
Python 스크립트는 stderr 에 `progress 0..100` 을 출력하고, 그 외 stderr 줄은 잡 로그에 쌓인다.
빈 구간 확인: `pyenv/venv/bin/python bin/audit.py --audio writable/media/N/audio.wav --segments writable/media/N/segments.json`
기존 결과물 타이밍만 재정렬: `pyenv/venv/bin/python bin/retime.py --dir writable/media/N` (ko/SDH 트랙도 함께 이동)

## 스택

| 구성 | 버전 |
|---|---|
| Ubuntu | 26.04 LTS (WSL2) |
| nginx / PHP-FPM / MySQL | nginx 1.28, PHP 8.5-FPM, MySQL 8.4 LTS |
| CodeIgniter | 4.7.x |
| Python | uv 로 받은 3.13 venv `pyenv/venv` (faster-whisper, whisperx 정렬, easyocr), 서버 python3 는 3.14 |
| Whisper | faster-whisper large-v3 + kotoba-tech/kotoba-whisper-v2.0-faster (일본어 보충) |
| Ollama | `huihui_ai/qwen3-vl-abliterated:8b-instruct` (장면/OCR), `huihui_ai/Qwen3.6-abliterated:35b-a3b-q4_K` (번역, 23 GB, CPU/GPU 분할) |

VRAM 10 GB 기준: Whisper large-v3 float16 약 3 GB, Qwen3-VL-8B Q4 약 6 GB, 35B-A3B 는 RAM 위주. 모델은 단계별로 순차 사용하고
Ollama 는 `OLLAMA_KEEP_ALIVE=0` + translate 종료 시 언로드로 다음 잡의 Whisper 와 겹치지 않게 한다.

## 설치

```bash
sudo apt update && sudo apt install -y nginx git unzip curl ffmpeg mysql-server \
  php-fpm php-cli php-mysql php-intl php-mbstring php-xml php-curl php-zip \
  python3 python3-pip
curl -sS https://getcomposer.org/installer | php && sudo mv composer.phar /usr/local/bin/composer

sudo mkdir -p /var/www && sudo chown $USER:www-data /var/www && sudo chmod 775 /var/www
cd /var/www && git clone git@github.com:Ted197905/MovTrans.git movtrans && cd movtrans
composer install --no-dev
cp env .env          # baseURL, DB 계정 편집
php spark key:generate
```

DB:

```sql
CREATE DATABASE movtrans CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE USER 'movtrans'@'localhost' IDENTIFIED BY '...';
GRANT ALL PRIVILEGES ON movtrans.* TO 'movtrans'@'localhost';
```

```bash
php spark migrate
php spark admin:set admin@example.com '비밀번호'
sudo chown -R $USER:www-data . && sudo chmod -R 755 . && sudo chmod -R 775 writable
```

nginx:

```bash
sudo cp deploy/nginx-movtrans.conf /etc/nginx/sites-available/movtrans   # php-fpm 소켓 버전 확인
sudo ln -s /etc/nginx/sites-available/movtrans /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

php.ini(fpm): `upload_max_filesize`, `post_max_size` 를 32M 이상, `max_execution_time` 300.

### Python (faster-whisper, whisperx 정렬, easyocr)

서버 Python 이 3.14 라 whisperx(3.13 미만만 지원)를 설치할 수 없다. uv 로 Python 3.13 venv 를 `pyenv/` 에 만든다 (sudo 불필요, 시스템/MVCut 영향 없음).

```bash
cd /var/www/movtrans/pyenv
export UV_INSTALL_DIR=$PWD/bin UV_NO_MODIFY_PATH=1 UV_PYTHON_INSTALL_DIR=$PWD/python UV_CACHE_DIR=$PWD/.uv-cache
curl -LsSf https://astral.sh/uv/install.sh | sh
bin/uv python install 3.13
bin/uv venv --python 3.13 venv
bin/uv pip install --python venv/bin/python whisperx easyocr opencv-python-headless scipy
venv/bin/python -c "import torch, whisperx, easyocr; print(torch.cuda.is_available())"
```

모델 캐시(HF: faster-whisper-large-v3, kotoba-whisper, wav2vec2-ja 정렬; EasyOCR 검출 모델)는 워커 HOME 인 `writable/.cache` 에 첫 잡에서 받아진다.

### 화자 분리

```bash
# NVIDIA Sortformer (주): 별도 venv, nvidia/diar_streaming_sortformer_4spk-v2 (CC-BY-4.0, 약관 동의 불필요)
bin/uv venv --python 3.12 nemo && bin/uv pip install --python nemo/bin/python Cython packaging "nemo_toolkit[asr]"
# pyannote 3.1 (교차검증): HF 계정에서 pyannote/segmentation-3.0, speaker-diarization-3.1, speaker-diarization-community-1 약관 동의 후
# 읽기 토큰을 .env 에: movtrans.hfToken = hf_...
```
`.env` `movtrans.diarizer` = both(기본) | auto | sortformer | pyannote | pitch. 잡 로그에 두 결과의 단어 단위 일치율이 남는다.

`.env`: `movtrans.python = /var/www/movtrans/pyenv/venv/bin/python` (pylibs 폴더가 없으면 PYTHONPATH 는 설정되지 않는다).

### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull huihui_ai/qwen3-vl-abliterated:8b-instruct       # 장면 노트, 화면 텍스트 판독 (6 GB)
ollama pull huihui_ai/Qwen3.6-abliterated:35b-a3b-q4_K       # 번역 (23 GB, RAM 32 GB 필요)
```
`/etc/systemd/system/ollama.service.d/override.conf` 에 `Environment=OLLAMA_KEEP_ALIVE=0`.

다른 모델을 쓰려면 `.env` 에 `movtrans.vlModel = ...`.

### 워커

```bash
sudo cp deploy/movtrans-worker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now movtrans-worker
journalctl -u movtrans-worker -f
```

WSL 에서 systemd 가 꺼져 있으면 `/etc/wsl.conf` 에 `[boot] systemd=true` 를 넣고 `wsl --shutdown`.

## 배포 (STUDIO WSL)

MVCut과 동일: Windows `MovTrans/ServerCode` 가 git 트리, `MovTrans/deploy.sh` 와 `MovTrans/.deploy/setup-ssh.sh` 는 트리 밖.
Cowork device_bash 에서 `bash .deploy/setup-ssh.sh` 후 `./deploy.sh "메시지"`. rsync 는 `pylibs`, `pyenv`, `models`, `writable`, `.env` 를 제외한다
(서버 전용, `--delete` 로 지워지면 안 됨). 서버 경로 `/var/www/movtrans`, 소유자 `kaiseian:www-data`.

## 개발

```bash
php spark serve
php spark worker:run --once   # 잡 하나 처리 후 종료
```

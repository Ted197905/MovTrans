# MovTrans

영상을 업로드하면 원어 자막(WhisperX)과 장면 이해 기반 한국어 자막(Qwen3-VL)을 만들어 주는 웹 서비스.
단일 관리자 계정, CodeIgniter 4 + MySQL, 무거운 처리는 systemd 워커가 담당한다.

## 흐름

1. 업로드 (청크 8 MB, 최대 8 GB) + 원어 선택 -> `videos` 행과 `jobs` 행 생성
2. 워커 `php spark worker:run` 이 `jobs` 를 2초마다 폴링해 단계별로 실행
   - `prepare`   ffmpeg 16 kHz 오디오 추출, 브라우저가 못 재생하는 코덱/컨테이너면 720p H.264 프록시
   - `transcribe` `bin/transcribe.py` WhisperX 전사 + 정렬 -> `segments.json`, `orig.srt/vtt`
   - `scene`     `bin/scene.py` 세그먼트마다 프레임 1장을 Qwen3-VL(Ollama)로 분석 -> `scenes.json`
   - `translate` `bin/translate.py` 영상 개요 + 장면 노트 + 이전 문맥으로 15줄씩 번역 -> `ko.srt/vtt`
3. 진행 페이지가 `/api/jobs/{id}` 를 폴링해 단계별 진행 바 표시, 완료되면 `<video>` + `<track>` 로 재생

파일은 `writable/media/{id}/` 에 두고 nginx `X-Accel-Redirect` 로 전송한다.
Python 스크립트는 stderr 에 `progress 0..100` 을 출력하고, 그 외 stderr 줄은 잡 로그에 쌓인다.

## 스택

| 구성 | 버전 |
|---|---|
| Ubuntu | 26.04 LTS (WSL2) |
| nginx / PHP-FPM / MySQL | nginx 1.28, PHP 8.5-FPM, MySQL 8.4 LTS |
| CodeIgniter | 4.7.x |
| WhisperX | 최신 (large-v3) |
| Ollama | 최신, `huihui_ai/qwen3-vl-abliterated:8b-instruct` |

VRAM 10 GB 기준: WhisperX large-v3 float16 약 5 GB, Qwen3-VL-8B Q4 약 6 GB. 두 모델을 순차로 쓰고,
translate 종료 시 Ollama 모델을 언로드(`keep_alive 0`)해서 다음 잡의 WhisperX 와 겹치지 않게 한다.

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

### Python (WhisperX)

```bash
pip install --target ./pylibs --index-url https://download.pytorch.org/whl/cu126 torch torchaudio
pip install --target ./pylibs -r bin/requirements.txt
PYTHONPATH=./pylibs python3 -c "import whisperx; print('ok')"
```

`pylibs` 가 있으면 워커가 자동으로 `PYTHONPATH` 에 넣는다. venv 를 쓰려면 `.env` 에 `movtrans.python = /path/venv/bin/python` 을 지정하고 `movtrans.pylibs =` 로 비운다.

### Ollama (Qwen3-VL)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull huihui_ai/qwen3-vl-abliterated:8b-instruct
```

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
Cowork device_bash 에서 `bash .deploy/setup-ssh.sh` 후 `./deploy.sh "메시지"`. rsync 는 `pylibs`, `models`, `writable`, `.env` 를 제외한다
(서버 전용, `--delete` 로 지워지면 안 됨). 서버 경로 `/var/www/movtrans`, 소유자 `kaiseian:www-data`.

## 개발

```bash
php spark serve
php spark worker:run --once   # 잡 하나 처리 후 종료
```

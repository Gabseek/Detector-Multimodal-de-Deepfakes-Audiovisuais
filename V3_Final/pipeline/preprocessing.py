# Prepara os vídeos, o áudio e os quadros.
import json
import subprocess
from pathlib import Path


# Lê os dados do vídeo e prepara o áudio para a análise.
class VideoPreprocessor:

    def __init__(self, output_dir='output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_command(self, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Erro ao executar FFmpeg/FFprobe:\n{error.stderr}') from error
        except FileNotFoundError as error:
            raise RuntimeError('FFmpeg ou FFprobe não foi encontrado. Verifique se está instalado e disponível no PATH.') from error

    def get_video_info(self, video_path):
        video_path = Path(video_path)
        command = [
            'ffprobe',
            '-v',
            'error',
            '-show_streams',
            '-show_format',
            '-of',
            'json',
            str(video_path)
        ]
        result = self._run_command(command)
        return json.loads(result.stdout)

    @staticmethod
    def has_audio_stream(video_info):
        streams = video_info.get('streams', [])
        for stream in streams:
            if stream.get('codec_type') == 'audio':
                return True
        return False

    @staticmethod
    def has_video_stream(video_info):
        streams = video_info.get('streams', [])
        for stream in streams:
            if stream.get('codec_type') == 'video':
                return True
        return False

    def extract_audio(self, video_path):
        video_path = Path(video_path)
        audio_dir = self.output_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f'{video_path.stem}_16khz_mono.wav'
        command = [
            'ffmpeg',
            '-y',
            '-i',
            str(video_path),
            '-vn',
            '-ac',
            '1',
            '-ar',
            '16000',
            '-c:a',
            'pcm_s16le',
            str(audio_path)
        ]
        self._run_command(command)
        return audio_path

    def process(self, video_path):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f'Vídeo não encontrado: {video_path}')
        print('\nObtendo metadados com FFprobe...')
        video_info = self.get_video_info(video_path)
        has_video = self.has_video_stream(video_info)
        has_audio = self.has_audio_stream(video_info)
        if not has_video:
            raise RuntimeError('O arquivo informado não possui uma stream de vídeo válida.')
        if has_audio:
            print('Stream de áudio encontrada.')
            print('Extraindo áudio em WAV mono 16 kHz...')
            audio_path = self.extract_audio(video_path)
            print(f'Áudio extraído: {audio_path}')
        else:
            print('\nNenhuma stream de áudio foi encontrada no vídeo.')
            audio_path = None
        return {
            'original_video': video_path,
            'audio': audio_path,
            'has_audio': has_audio,
            'has_video': has_video,
            'video_info': video_info
        }


import cv2
from pathlib import Path


# Separa quadros do vídeo nos intervalos configurados.
class FrameExtractor:

    def __init__(self, output_dir='output/frames', interval_seconds=0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds

    def _clear_output(self):
        for file_path in self.output_dir.glob('frame_*.jpg'):
            file_path.unlink(missing_ok=True)

    def extract_frames(self, video_path):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f'Vídeo não encontrado: {video_path}')
        self._clear_output()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f'Não foi possível abrir o vídeo: {video_path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        frame_interval = max(1, int(fps * self.interval_seconds))
        print('\nINFORMAÇÕES DO VÍDEO')
        print('-' * 50)
        print(f'FPS: {fps:.2f}')
        print(f'Total de frames: {total_frames}')
        print(f'Duração: {duration:.2f} segundos')
        print(f'Amostragem: 1 frame a cada {self.interval_seconds}s')
        print('-' * 50)
        extracted_frames = []
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                timestamp = frame_index / fps if fps > 0 else 0
                frame_filename = self.output_dir / f'frame_{frame_index:06d}.jpg'
                success_write = cv2.imwrite(str(frame_filename), frame)
                if success_write:
                    extracted_frames.append({
                        'frame_index': frame_index,
                        'timestamp': timestamp,
                        'path': frame_filename
                    })
            frame_index += 1
        capture.release()
        print(f'\nFrames extraídos: {len(extracted_frames)}')
        return {
            'frames': extracted_frames,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }

    def process(self, video_path):
        return self.extract_frames(video_path)

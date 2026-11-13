# Analisa a faixa sonora com o DF-Arena-1B.
import json
import math
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf


# O áudio é dividido em janelas antes de ser avaliado pelo modelo.
class DFArenaAudioDetector:
    SAMPLE_RATE = 16000
    WINDOW_SAMPLES = 64600
    SPOOF_INDEX = 0
    BONAFIDE_INDEX = 1

    def __init__(
        self,
        model_path='models/audio/df_arena_1b_int8.onnx',
        output_path='output/audio_score.json',
        threshold=0.5,
        suspicious_fraction=0.25
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.threshold = float(threshold)
        self.suspicious_fraction = float(suspicious_fraction)
        if not self.model_path.exists():
            raise FileNotFoundError(f'Modelo DF-Arena-1B não encontrado: {self.model_path}')
        if not 0 < self.suspicious_fraction <= 1:
            raise ValueError('suspicious_fraction deve estar entre 0 e 1.')
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print('\nCarregando DF-Arena-1B INT8...')
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print('DF-Arena-1B carregado com sucesso.')
        print(f'Entrada ONNX: {self.input_name}')
        print(f'Saída ONNX: {self.output_name}')

    def _load_audio(self, audio_path):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f'Áudio não encontrado: {audio_path}')
        audio, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError(f'O DF-Arena-1B espera áudio em {self.SAMPLE_RATE} Hz, mas recebeu {sample_rate} Hz.')
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError('O arquivo de áudio está vazio.')
        return (audio, sample_rate)

    def _repeat_to_window(self, audio):
        if len(audio) == 0:
            raise ValueError('Segmento de áudio vazio.')
        if len(audio) >= self.WINDOW_SAMPLES:
            return audio[:self.WINDOW_SAMPLES].astype(np.float32)
        repetitions = math.ceil(self.WINDOW_SAMPLES / len(audio))
        padded = np.tile(audio, repetitions)[:self.WINDOW_SAMPLES]
        return np.asarray(padded, dtype=np.float32)

    def _create_segments(self, audio):
        total_samples = len(audio)
        if total_samples <= self.WINDOW_SAMPLES:
            fixed_audio = self._repeat_to_window(audio)
            return [{
                'index': 0,
                'start_sample': 0,
                'end_sample': total_samples,
                'start_time': 0.0,
                'end_time': float(total_samples / self.SAMPLE_RATE),
                'was_repeated': total_samples < self.WINDOW_SAMPLES,
                'audio': fixed_audio
            }]
        starts = list(range(0, total_samples - self.WINDOW_SAMPLES + 1, self.WINDOW_SAMPLES))
        last_end = starts[-1] + self.WINDOW_SAMPLES
        if last_end < total_samples:
            final_start = total_samples - self.WINDOW_SAMPLES
            if final_start != starts[-1]:
                starts.append(final_start)
        segments = []
        for segment_index, start_sample in enumerate(starts):
            end_sample = start_sample + self.WINDOW_SAMPLES
            segment_audio = audio[start_sample:end_sample].astype(np.float32)
            if len(segment_audio) != self.WINDOW_SAMPLES:
                raise RuntimeError('Janela criada com tamanho incorreto.')
            segments.append({
                'index': segment_index,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'start_time': float(start_sample / self.SAMPLE_RATE),
                'end_time': float(end_sample / self.SAMPLE_RATE),
                'was_repeated': False,
                'audio': segment_audio
            })
        return segments

    @staticmethod
    def _softmax(logits):
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - np.max(logits)
        exponential = np.exp(logits)
        probabilities = exponential / np.sum(exponential)
        return probabilities

    def _predict_segment(self, segment_audio):
        model_input = segment_audio[np.newaxis, :].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: model_input})
        logits = np.asarray(outputs[0]).reshape(-1)
        if logits.size != 2:
            raise RuntimeError(f'Saída inesperada do DF-Arena-1B. Esperados 2 logits, recebidos {logits.size}.')
        probabilities = self._softmax(logits)
        spoof_score = float(probabilities[self.SPOOF_INDEX])
        bonafide_score = float(probabilities[self.BONAFIDE_INDEX])
        if spoof_score >= self.threshold:
            classification = 'SINTETICO'
            predicted_label = 1
        else:
            classification = 'AUTENTICO'
            predicted_label = 0
        return {
            'spoof_logit': float(logits[self.SPOOF_INDEX]),
            'bonafide_logit': float(logits[self.BONAFIDE_INDEX]),
            'spoof_score': spoof_score,
            'bonafide_score': bonafide_score,
            'classification': classification,
            'label_predito': predicted_label
        }

    # Os segmentos mais suspeitos contribuem para o score final.
    def _aggregate_segments(self, segment_results):
        if not segment_results:
            raise RuntimeError('Nenhum segmento foi analisado.')
        count = max(1, math.ceil(len(segment_results) * self.suspicious_fraction))
        ordered = sorted(segment_results, key=lambda item: item['spoof_score'], reverse=True)
        selected = ordered[:count]
        spoof_score = float(np.mean([item['spoof_score'] for item in selected]))
        bonafide_score = float(np.mean([item['bonafide_score'] for item in selected]))
        if spoof_score >= self.threshold:
            classification = 'SINTETICO'
            predicted_label = 1
        else:
            classification = 'AUTENTICO'
            predicted_label = 0
        return {
            'spoof_score': spoof_score,
            'bonafide_score': bonafide_score,
            'classification': classification,
            'label_predito': predicted_label,
            'selected_segments': selected,
            'selected_count': count
        }

    def process(self, audio_path):
        print('\n')
        print('=' * 70)
        print('DF-ARENA-1B - ANÁLISE DO ÁUDIO COMPLETO')
        print('=' * 70)
        audio, sample_rate = self._load_audio(audio_path)
        total_samples = len(audio)
        total_duration = total_samples / sample_rate
        print(f'\nÁudio: {audio_path}')
        print(f'Duração total: {total_duration:.2f}s')
        print(f'Sample rate: {sample_rate} Hz')
        print(f'Janela: {self.WINDOW_SAMPLES} amostras')
        print(f'Duração da janela: {self.WINDOW_SAMPLES / self.SAMPLE_RATE:.2f}s')
        segments = self._create_segments(audio)
        print(f'Janelas a analisar: {len(segments)}')
        segment_results = []
        print('\n')
        print('RESULTADOS POR JANELA')
        print('-' * 70)
        for segment in segments:
            prediction = self._predict_segment(segment['audio'])
            result = {
                'segment_index': segment['index'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'was_repeated': segment['was_repeated'],
                'spoof_logit': prediction['spoof_logit'],
                'bonafide_logit': prediction['bonafide_logit'],
                'spoof_score': prediction['spoof_score'],
                'bonafide_score': prediction['bonafide_score'],
                'classification': prediction['classification'],
                'label_predito': prediction['label_predito']
            }
            segment_results.append(result)
            print(f"Janela {result['segment_index'] + 1:03d} | {result['start_time']:.2f}s - {result['end_time']:.2f}s | spoof: {result['spoof_score']:.4f} | bona fide: {result['bonafide_score']:.4f} | {result['classification']}")
        aggregated = self._aggregate_segments(segment_results)
        spoof_score = aggregated['spoof_score']
        bonafide_score = aggregated['bonafide_score']
        classification = aggregated['classification']
        predicted_label = aggregated['label_predito']
        selected_segments = aggregated['selected_segments']
        most_suspicious = max(segment_results, key=lambda item: item['spoof_score'])
        result = {
            'model': 'DF-Arena-1B INT8',
            'source_model': 'DF-Arena-1B',
            'audio_path': str(audio_path),
            'sample_rate': sample_rate,
            'total_samples': total_samples,
            'total_duration_seconds': float(total_duration),
            'window_samples': self.WINDOW_SAMPLES,
            'window_duration_seconds': float(self.WINDOW_SAMPLES / self.SAMPLE_RATE),
            'segments_analyzed': len(segment_results),
            'aggregation_method': 'top25_mean',
            'aggregation_fraction': self.suspicious_fraction,
            'segments_used_in_aggregation': len(selected_segments),
            'audio_score': spoof_score,
            'spoof_score': spoof_score,
            'bonafide_score': bonafide_score,
            'threshold': self.threshold,
            'classification': classification,
            'label_predito': predicted_label,
            'most_suspicious_segment': {
                'segment_index': most_suspicious['segment_index'],
                'start_time': most_suspicious['start_time'],
                'end_time': most_suspicious['end_time'],
                'spoof_score': most_suspicious['spoof_score'],
                'bonafide_score': most_suspicious['bonafide_score'],
                'classification': most_suspicious['classification']
            },
            'selected_segments': selected_segments,
            'segment_scores': segment_results
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
        print('\n')
        print('=' * 70)
        print('RESULTADO FINAL DO ÁUDIO')
        print('=' * 70)
        print(f'\nScore spoof agregado: {spoof_score:.4f}')
        print(f'Score bona fide agregado: {bonafide_score:.4f}')
        print(f'Threshold: {self.threshold:.2f}')
        print(f'Classificação: {classification}')
        print(f'Agregação: top25_mean')
        print('\nTrecho mais suspeito:')
        print(f"{most_suspicious['start_time']:.2f}s - {most_suspicious['end_time']:.2f}s")
        print(f"Score spoof: {most_suspicious['spoof_score']:.4f}")
        return result


import onnxruntime as ort


# Aplica ao detector os limites de threads do perfil selecionado.
class ThreadAwareDFArenaAudioDetector(DFArenaAudioDetector):

    def __init__(
        self,
        model_path='models/audio/df_arena_1b_int8.onnx',
        output_path='output/audio_score.json',
        threshold=0.5,
        suspicious_fraction=0.25,
        cpu_threads=1
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.threshold = float(threshold)
        self.suspicious_fraction = float(suspicious_fraction)
        self.cpu_threads = max(1, int(cpu_threads))
        if not self.model_path.exists():
            raise FileNotFoundError(f'Modelo DF-Arena-1B não encontrado: {self.model_path}')
        if not 0 < self.suspicious_fraction <= 1:
            raise ValueError('suspicious_fraction deve estar entre 0 e 1.')
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = self.cpu_threads
        session_options.inter_op_num_threads = 1
        try:
            session_options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        except Exception:
            pass
        print('\nCarregando DF-Arena-1B INT8...')
        print(f'Threads internas ONNX: {self.cpu_threads}')
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print('DF-Arena-1B carregado com sucesso.')
        print(f'Entrada ONNX: {self.input_name}')
        print(f'Saída ONNX: {self.output_name}')

# Cuida dos recursos, tempos e relatórios dos testes.
import csv
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


# Controla quantas tarefas e threads podem usar o processador.
class ResourceManager:
    MODES = {'SEQUENCIAL', 'ECONOMICO', 'BALANCEADO', 'DESEMPENHO'}

    def __init__(self, mode='BALANCEADO'):
        self.mode = str(mode or 'BALANCEADO').strip().upper()
        if self.mode not in self.MODES:
            raise ValueError(f"Modo de desempenho inválido: {self.mode}. Use: {', '.join(sorted(self.MODES))}.")
        self.logical_cpus = max(1, os.cpu_count() or 1)
        self._calculate_limits()

    def _calculate_limits(self):
        n = self.logical_cpus
        if self.mode == 'SEQUENCIAL':
            self.heavy_workers = 1
            self.reserved_cpus = max(1, math.ceil(n * 0.25)) if n > 2 else 0
        elif self.mode == 'ECONOMICO':
            self.heavy_workers = 1
            self.reserved_cpus = max(1, math.ceil(n * 0.5)) if n > 1 else 0
        elif self.mode == 'DESEMPENHO':
            if n >= 8:
                self.heavy_workers = 3
            elif n >= 4:
                self.heavy_workers = 2
            else:
                self.heavy_workers = 1
            self.reserved_cpus = 1 if n > 2 else 0
        else:
            self.heavy_workers = 2 if n >= 4 else 1
            self.reserved_cpus = max(1, math.ceil(n * 0.25)) if n > 2 else 0
        self.usable_cpus = max(1, n - self.reserved_cpus)
        self.cpu_threads_per_heavy_task = max(1, self.usable_cpus // self.heavy_workers)
        if self.mode == 'ECONOMICO':
            self.cpu_threads_per_heavy_task = min(self.cpu_threads_per_heavy_task, 4)
        elif self.mode != 'DESEMPENHO':
            self.cpu_threads_per_heavy_task = min(self.cpu_threads_per_heavy_task, 6)

    def apply_environment_limits(self):
        value = str(self.cpu_threads_per_heavy_task)
        os.environ['OMP_NUM_THREADS'] = value
        os.environ['MKL_NUM_THREADS'] = value
        os.environ['OPENBLAS_NUM_THREADS'] = value
        os.environ['NUMEXPR_NUM_THREADS'] = value
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def apply_runtime_limits(self):
        try:
            import torch
            torch.set_num_threads(self.cpu_threads_per_heavy_task)
        except Exception:
            pass
        try:
            import cv2
            cv2.setNumThreads(self.cpu_threads_per_heavy_task)
        except Exception:
            pass

    def as_dict(self):
        return {
            'mode': self.mode,
            'logical_cpus_detected': self.logical_cpus,
            'reserved_cpus_for_system': self.reserved_cpus,
            'usable_cpus_budget': self.usable_cpus,
            'max_parallel_heavy_tasks': self.heavy_workers,
            'cpu_threads_per_heavy_task': self.cpu_threads_per_heavy_task,
            'parallel_branches_enabled': self.heavy_workers > 1,
            'execution_profile': 'SEQUENTIAL_REFERENCE' if self.mode == 'SEQUENCIAL' else 'RESOURCE_SAVING' if self.mode == 'ECONOMICO' else 'BALANCED_PARALLEL' if self.mode == 'BALANCEADO' else 'PERFORMANCE_PARALLEL'
        }

    def print_summary(self):
        print('\n' + '=' * 70)
        print('CONTROLE DE RECURSOS')
        print('=' * 70)
        print(f'Modo: {self.mode}')
        print(f'CPUs lógicas detectadas: {self.logical_cpus}')
        print(f'CPUs reservadas ao sistema: {self.reserved_cpus}')
        print(f'Orçamento do pipeline: {self.usable_cpus}')
        print(f'Tarefas pesadas simultâneas: {self.heavy_workers}')
        print('Execução entre ramos: ' + ('PARALELA CONTROLADA' if self.heavy_workers > 1 else 'SEQUENCIAL'))
        print(f'Threads internas por tarefa pesada: {self.cpu_threads_per_heavy_task}')
        print('=' * 70)


# Registra os tempos das etapas, inclusive quando há execução paralela.
class PerformanceMonitor:

    def __init__(self, resource_config=None):
        self.resource_config = resource_config or {}
        self._lock = threading.Lock()
        self._records = []
        self._application_start = None
        self._application_end = None
        self._sequence = 0

    def start_application(self):
        self._application_start = time.perf_counter()

    def finish_application(self):
        self._application_end = time.perf_counter()

    @contextmanager
    def measure(self, key, label, category, count_in_sequential_sum=True, metadata=None):
        start = time.perf_counter()
        error = None
        try:
            yield
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            ended_at = time.perf_counter()
            duration = ended_at - start
            self.record(
                key=key,
                label=label,
                category=category,
                seconds=duration,
                count_in_sequential_sum=count_in_sequential_sum,
                metadata=metadata,
                error=error,
                started_at=start,
                ended_at=ended_at
            )

    def record(
        self,
        key,
        label,
        category,
        seconds,
        count_in_sequential_sum=True,
        metadata=None,
        error=None,
        started_at=None,
        ended_at=None
    ):
        with self._lock:
            self._sequence += 1
            self._records.append({
                'sequence': self._sequence,
                'key': key,
                'label': label,
                'category': category,
                'seconds': float(seconds),
                'count_in_sequential_sum': bool(count_in_sequential_sum),
                'thread': threading.current_thread().name,
                'metadata': metadata or {},
                'error': error,
                'start_offset_seconds': float(started_at - self._application_start) if started_at is not None and self._application_start is not None else None,
                'end_offset_seconds': float(ended_at - self._application_start) if ended_at is not None and self._application_start is not None else None
            })

    def record_cached(self, key, label, category):
        self.record(
            key=key,
            label=label,
            category=category,
            seconds=0.0,
            count_in_sequential_sum=False,
            metadata={'cached_in_memory': True}
        )

    @property
    def total_wall_seconds(self):
        if self._application_start is None:
            return None
        end = self._application_end or time.perf_counter()
        return float(end - self._application_start)

    def _category_leaf_totals(self):
        totals = {}
        for item in self._records:
            if not item['count_in_sequential_sum']:
                continue
            totals.setdefault(item['category'], 0.0)
            totals[item['category']] += item['seconds']
        return totals

    def build_report(self):
        wall = self.total_wall_seconds or 0.0
        leaf_records = [item for item in self._records if item['count_in_sequential_sum']]
        sequential_sum = sum((item['seconds'] for item in leaf_records))
        overlap = max(0.0, sequential_sum - wall)
        overlap_factor = sequential_sum / wall if wall > 0 else 0.0
        branch_records = [item for item in self._records if not item['count_in_sequential_sum'] and item['key'].startswith('branch.')]
        intervalos = sorted(((
            r['start_offset_seconds'],
            r['end_offset_seconds']
        ) for r in leaf_records if r.get('start_offset_seconds') is not None and r.get('end_offset_seconds') is not None))
        segmentos = []
        for inicio, fim in intervalos:
            if segmentos and inicio <= segmentos[-1][1]:
                segmentos[-1] = (segmentos[-1][0], max(fim, segmentos[-1][1]))
            else:
                segmentos.append((inicio, fim))
        atividade_medida = sum((fim - inicio for inicio, fim in segmentos))
        soma_intervalos = sum((fim - inicio for inicio, fim in intervalos))
        sobreposicao_medida = max(0.0, soma_intervalos - atividade_medida)
        return {
            'measured_parallel_overlap_seconds': sobreposicao_medida,
            'measured_active_wall_seconds': atividade_medida,
            'unmeasured_wall_seconds': max(0.0, wall - atividade_medida),
            'measurement_method': 'REGISTRO_TEMPOS_POR_ETAPA',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'resource_configuration': self.resource_config,
            'application_wall_time_seconds': wall,
            'sum_of_leaf_task_times_seconds': sequential_sum,
            'estimated_parallel_overlap_seconds': overlap,
            'task_time_to_wall_time_ratio': overlap_factor,
            'category_leaf_totals_seconds': self._category_leaf_totals(),
            'branch_wall_times_seconds': {item['key']: item['seconds'] for item in branch_records},
            'records': sorted(self._records, key=lambda item: item['sequence'])
        }

    def print_report(self):
        report = self.build_report()
        print('\n' + '=' * 78)
        print('RELATÓRIO DE TEMPO DE EXECUÇÃO')
        print('=' * 78)
        category_order = ['PRE_PROCESSAMENTO', 'VISUAL', 'AUDIO', 'FUSAO']
        records = report['records']
        for category in category_order:
            items = [item for item in records if item['category'] == category and item['count_in_sequential_sum']]
            if not items:
                continue
            print('\n' + category.replace('_', ' '))
            print('-' * 78)
            for item in items:
                print(f"  {item['label']:<46} {item['seconds']:>9.3f} s")
            subtotal = report['category_leaf_totals_seconds'].get(category, 0.0)
            print(f"  {'Soma das ferramentas da categoria':<46} {subtotal:>9.3f} s")
        branches = report['branch_wall_times_seconds']
        if branches:
            print('\nTEMPO DE PAREDE DOS RAMOS')
            print('-' * 78)
            labels = {
                'branch.visual': 'Ramo visual completo',
                'branch.audio': 'Ramo de detecção de áudio'
            }
            for key, seconds in branches.items():
                print(f'  {labels.get(key, key):<46} {seconds:>9.3f} s')
        print('\nRESUMO GLOBAL')
        print('-' * 78)
        print(f"  {'Tempo total real da aplicação':<46} {report['application_wall_time_seconds']:>9.3f} s")
        print(f"  {'Soma dos tempos das tarefas individuais':<46} {report['sum_of_leaf_task_times_seconds']:>9.3f} s")
        print(f"  {'Sobreposição estimada por paralelismo':<46} {report['estimated_parallel_overlap_seconds']:>9.3f} s")
        print(f"  {'Razão soma_tarefas / tempo_real':<46} {report['task_time_to_wall_time_ratio']:>9.3f} x")
        print('=' * 78)

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.build_report()
        json_path = output_dir / 'performance_report.json'
        txt_path = output_dir / 'performance_report.txt'
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=4, ensure_ascii=False)
        lines = [
            'RELATÓRIO DE DESEMPENHO DO PIPELINE',
            '=' * 72,
            f"Modo: {self.resource_config.get('mode')}",
            f"Tempo total real: {report['application_wall_time_seconds']:.3f} s",
            f"Soma das tarefas individuais: {report['sum_of_leaf_task_times_seconds']:.3f} s",
            f"Sobreposição estimada: {report['estimated_parallel_overlap_seconds']:.3f} s",
            '',
            'ETAPAS',
            '-' * 72
        ]
        for item in report['records']:
            if item['count_in_sequential_sum']:
                lines.append(f"[{item['category']}] {item['label']}: {item['seconds']:.3f} s (thread={item['thread']})")
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))
        report['json_path'] = str(json_path)
        report['txt_path'] = str(txt_path)
        return report


# Salva os tempos registrados para comparação entre testes.
def save_dataset_performance_csv(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'tempos_pipeline.csv'
    fields = [
        'video',
        'mode',
        'status',
        'erro',
        'application_wall_time_seconds',
        'sum_of_leaf_task_times_seconds',
        'estimated_parallel_overlap_seconds',
        'measured_parallel_overlap_seconds',
        'measured_active_wall_seconds',
        'unmeasured_wall_seconds',
        'task_time_to_wall_time_ratio',
        'pre_processamento_seconds',
        'visual_seconds',
        'audio_seconds',
        'fusao_seconds'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            performance = result.get('performance') or {}
            totals = performance.get('category_leaf_totals_seconds', {})
            config = performance.get('resource_configuration', {})
            writer.writerow({
                'video': str(result.get('video', '')),
                'mode': config.get('mode'),
                'status': result.get('status', 'OK'),
                'erro': result.get('erro', ''),
                'application_wall_time_seconds': performance.get('application_wall_time_seconds'),
                'sum_of_leaf_task_times_seconds': performance.get('sum_of_leaf_task_times_seconds'),
                'estimated_parallel_overlap_seconds': performance.get('estimated_parallel_overlap_seconds'),
                'measured_parallel_overlap_seconds': performance.get('measured_parallel_overlap_seconds'),
                'measured_active_wall_seconds': performance.get('measured_active_wall_seconds'),
                'unmeasured_wall_seconds': performance.get('unmeasured_wall_seconds'),
                'task_time_to_wall_time_ratio': performance.get('task_time_to_wall_time_ratio'),
                'pre_processamento_seconds': totals.get('PRE_PROCESSAMENTO'),
                'visual_seconds': totals.get('VISUAL'),
                'audio_seconds': totals.get('AUDIO'),
                'fusao_seconds': totals.get('FUSAO')
            })
    return csv_path


import csv
import json
import math
from pathlib import Path


# Organiza os vídeos do conjunto e consolida os relatórios.
class GenericDatasetProcessor:
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg'}

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f'Pasta não encontrada: {self.dataset_path}')
        if not self.dataset_path.is_dir():
            raise NotADirectoryError(f'O caminho informado não é uma pasta: {self.dataset_path}')
        self.labels = self._load_labels()

    @staticmethod
    def _normalizar_classe_real(label):
        value = str(label or '').strip().upper()
        if value in {'FAKE', 'FALSO', 'MANIPULADO', 'SUSPEITO'}:
            return 'SUSPEITO'
        if value in {
            'REAL',
            'VERDADEIRO',
            'AUTENTICO',
            'AUTÊNTICO',
            'NAO_SUSPEITO',
            'NÃO_SUSPEITO'
        }:
            return 'NAO_SUSPEITO'
        raise ValueError(f'Classe inválida em labels.csv: {label!r}. Use FAKE ou REAL.')

    def _load_labels(self):
        labels_path = self.dataset_path / 'labels.csv'
        if not labels_path.is_file():
            return {}
        labels = {}
        with labels_path.open(encoding='utf-8-sig', newline='') as arquivo:
            leitor = csv.DictReader(arquivo)
            if not leitor.fieldnames or not {'video', 'label'}.issubset(set(leitor.fieldnames)):
                raise ValueError('labels.csv deve conter as colunas video,label.')
            for row in leitor:
                video = str(row.get('video') or '').strip().replace('\\', '/')
                if not video:
                    raise ValueError('labels.csv contém um caminho de vídeo vazio.')
                key = Path(video).as_posix().casefold()
                if key in labels:
                    raise ValueError(f'Vídeo duplicado em labels.csv: {video}')
                labels[key] = self._normalizar_classe_real(row.get('label'))
        return labels

    def get_label(self, relative_path):
        key = Path(relative_path).as_posix().casefold()
        if key in self.labels:
            return self.labels[key]
        partes = {p.casefold() for p in Path(relative_path).parts[:-1]}
        fake = bool(partes & {'fake', 'fakes', 'falso', 'falsos'})
        real = bool(partes & {'real', 'reais', 'verdadeiro', 'verdadeiros'})
        if fake != real:
            return 'SUSPEITO' if fake else 'NAO_SUSPEITO'
        return None

    def find_videos(self):
        videos = [arquivo for arquivo in self.dataset_path.rglob('*') if arquivo.is_file() and arquivo.suffix.lower() in self.VIDEO_EXTENSIONS]
        videos.sort(key=lambda caminho: str(caminho).lower())
        return videos

    def get_relative_path(self, video_path):
        return Path(video_path).relative_to(self.dataset_path)

    @staticmethod

    # Calcula os indicadores usando os rótulos disponíveis.
    def create_summary(results):

        def contar(campo, valor):
            return sum((item.get(campo) == valor for item in results))
        avaliados_visuais = [item for item in results if item.get('classe_real') in {'SUSPEITO', 'NAO_SUSPEITO'} and item.get('classificacao_risco_visual') in {'SUSPEITO', 'NAO_SUSPEITO'} and (item.get('status') in {'OK', 'OK_VISUAL_ONLY'})]
        verdadeiros_suspeitos = sum((item['classe_real'] == 'SUSPEITO' and item['classificacao_risco_visual'] == 'SUSPEITO' for item in avaliados_visuais))
        falsos_nao_suspeitos = sum((item['classe_real'] == 'SUSPEITO' and item['classificacao_risco_visual'] == 'NAO_SUSPEITO' for item in avaliados_visuais))
        verdadeiros_nao_suspeitos = sum((item['classe_real'] == 'NAO_SUSPEITO' and item['classificacao_risco_visual'] == 'NAO_SUSPEITO' for item in avaliados_visuais))
        falsos_suspeitos = sum((item['classe_real'] == 'NAO_SUSPEITO' and item['classificacao_risco_visual'] == 'SUSPEITO' for item in avaliados_visuais))
        total_rotulados_fake = verdadeiros_suspeitos + falsos_nao_suspeitos
        total_rotulados_real = verdadeiros_nao_suspeitos + falsos_suspeitos
        total_previstos_suspeitos = verdadeiros_suspeitos + falsos_suspeitos
        total_rotulados = len(avaliados_visuais)
        sensibilidade_fake = verdadeiros_suspeitos / total_rotulados_fake if total_rotulados_fake else None
        especificidade_real = verdadeiros_nao_suspeitos / total_rotulados_real if total_rotulados_real else None
        scores_fake = []
        scores_real = []
        for item in avaliados_visuais:
            try:
                score = float(item.get('score_visual'))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            if item['classe_real'] == 'SUSPEITO':
                scores_fake.append(score)
            else:
                scores_real.append(score)
        roc_auc_visual = None
        if scores_fake and scores_real:
            roc_auc_visual = sum((1.0 if fake_score > real_score else 0.5 if fake_score == real_score else 0.0 for fake_score in scores_fake for real_score in scores_real)) / (len(scores_fake) * len(scores_real))

        def metricas_rotuladas(campo_classe, campo_score):
            validos = [item for item in results if item.get('status') in {'OK', 'OK_VISUAL_ONLY'} and item.get('classe_real') in {'SUSPEITO', 'NAO_SUSPEITO'} and (item.get(campo_classe) in {'SUSPEITO', 'NAO_SUSPEITO'})]
            tp = sum((x['classe_real'] == 'SUSPEITO' and x[campo_classe] == 'SUSPEITO' for x in validos))
            fn = sum((x['classe_real'] == 'SUSPEITO' and x[campo_classe] == 'NAO_SUSPEITO' for x in validos))
            tn = sum((x['classe_real'] == 'NAO_SUSPEITO' and x[campo_classe] == 'NAO_SUSPEITO' for x in validos))
            fp = sum((x['classe_real'] == 'NAO_SUSPEITO' and x[campo_classe] == 'SUSPEITO' for x in validos))
            sens = tp / (tp + fn) if tp + fn else None
            esp = tn / (tn + fp) if tn + fp else None
            scores = {'SUSPEITO': [], 'NAO_SUSPEITO': []}
            for x in validos:
                try:
                    sc = float(x.get(campo_score))
                    if math.isfinite(sc):
                        scores[x['classe_real']].append(sc)
                except (TypeError, ValueError):
                    pass
            positivos, negativos = (scores['SUSPEITO'], scores['NAO_SUSPEITO'])
            auc = sum((1 if p > n else 0.5 if p == n else 0 for p in positivos for n in negativos)) / (len(positivos) * len(negativos)) if positivos and negativos else None
            return {
                'videos_avaliados': len(validos),
                'acuracia': (tp + tn) / len(validos) if validos else None,
                'sensibilidade_fake': sens,
                'especificidade_real': esp,
                'balanced_accuracy': (sens + esp) / 2 if sens is not None and esp is not None else None,
                'precisao_suspeito': tp / (tp + fp) if tp + fp else None,
                'f1_suspeito': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                'roc_auc': auc,
                'matriz_confusao': {
                    'fake_classificado_suspeito': tp,
                    'fake_classificado_nao_suspeito': fn,
                    'real_classificado_nao_suspeito': tn,
                    'real_classificado_suspeito': fp
                }
            }
        return {
            'total_videos': len(results),
            'processados_com_sucesso': sum((item.get('status') in {'OK', 'OK_VISUAL_ONLY'} for item in results)),
            'ignorados': sum((item.get('status') == 'IGNORADO_SEM_AUDIO' for item in results)),
            'erros': sum((str(item.get('status', '')).startswith('ERRO') for item in results)),
            'videos_com_audio': sum((item.get('possui_audio') is True for item in results)),
            'videos_sem_audio': sum((item.get('possui_audio') is False for item in results)),
            'multimodais_suspeitos': contar('classificacao_multimodal', 'SUSPEITO'),
            'multimodais_nao_suspeitos': contar('classificacao_multimodal', 'NAO_SUSPEITO'),
            'visuais_suspeitos': contar('classificacao_risco_visual', 'SUSPEITO'),
            'visuais_nao_suspeitos': contar('classificacao_risco_visual', 'NAO_SUSPEITO'),
            'audios_suspeitos': contar('classificacao_risco_audio', 'SUSPEITO'),
            'audios_nao_suspeitos': contar('classificacao_risco_audio', 'NAO_SUSPEITO'),
            'metricas_audio_rotuladas': metricas_rotuladas(
                'classificacao_risco_audio',
                'score_audio'
            ),
            'metricas_multimodais_rotuladas': metricas_rotuladas(
                'classificacao_multimodal',
                'score_suspeita_multimodal'
            ),
            'metricas_visuais_rotuladas': {
                'videos_avaliados': total_rotulados,
                'acuracia': (verdadeiros_suspeitos + verdadeiros_nao_suspeitos) / total_rotulados if total_rotulados else None,
                'sensibilidade_fake': sensibilidade_fake,
                'especificidade_real': especificidade_real,
                'balanced_accuracy': (sensibilidade_fake + especificidade_real) / 2.0 if sensibilidade_fake is not None and especificidade_real is not None else None,
                'roc_auc_visual': roc_auc_visual,
                'precisao_suspeito': verdadeiros_suspeitos / total_previstos_suspeitos if total_previstos_suspeitos else None,
                'f1_suspeito': 2 * verdadeiros_suspeitos / (2 * verdadeiros_suspeitos + falsos_suspeitos + falsos_nao_suspeitos) if 2 * verdadeiros_suspeitos + falsos_suspeitos + falsos_nao_suspeitos else None,
                'matriz_confusao': {
                    'fake_classificado_suspeito': verdadeiros_suspeitos,
                    'fake_classificado_nao_suspeito': falsos_nao_suspeitos,
                    'real_classificado_nao_suspeito': verdadeiros_nao_suspeitos,
                    'real_classificado_suspeito': falsos_suspeitos
                }
            }
        }

    def save_results(self, results, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'resultados_dataset.csv'
        campos = [
            'indice',
            'video',
            'caminho_relativo',
            'caminho_completo',
            'possui_audio',
            'modo',
            'classe_real',
            'correto_visual',
            'marca_dagua_confirmada',
            'marca_dagua_detectada',
            'etapa_visual_executada',
            'effort_executado',
            'status_triagem_marca',
            'modo_spsl',
            'spsl_interrompeu_effort',
            'spsl_atalho_elegivel',
            'spsl_frames_fortes',
            'status_spsl',
            'score_spsl',
            'score_medio_spsl',
            'score_top25_spsl',
            'classificacao_spsl_exploratoria',
            'frames_spsl',
            'tempo_spsl_segundos',
            'tempo_total_video_segundos',
            'score_effort_guided_full',
            'classificacao_effort_guided_full',
            'score_effort_guided_face',
            'classificacao_effort_guided_face',
            'faces_guiadas_selecionadas',
            'taxa_deteccao_face_scan',
            'correto_multimodal',
            'correto_audio',
            'score_visual',
            'classificacao_visual',
            'classificacao_risco_visual',
            'frames_analisados',
            'rostos_analisados',
            'score_audio',
            'bonafide_score',
            'classificacao_audio',
            'classificacao_risco_audio',
            'janelas_audio_analisadas',
            'trecho_audio_suspeito_inicio',
            'trecho_audio_suspeito_fim',
            'trecho_audio_spoof_score',
            'classificacao_multimodal',
            'score_suspeita_multimodal',
            'confianca_multimodal',
            'modalidades_usadas_fusao',
            'quantidade_modalidades_fusao',
            'frames_suspeitos_quantidade',
            'frames_suspeitos_resumo',
            'timestamps_suspeitos_quantidade',
            'timestamps_suspeitos_resumo',
            'arquivo_relatorio_multimodal_json',
            'arquivo_relatorio_multimodal_txt',
            'status',
            'diretorio_saida'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=campos)
            escritor.writeheader()
            for resultado in results:
                escritor.writerow({campo: resultado.get(campo) for campo in campos})
        resumo = self.create_summary(results)
        summary_path = output_dir / 'resumo_dataset.json'
        with open(summary_path, 'w', encoding='utf-8') as arquivo:
            json.dump(resumo, arquivo, indent=4, ensure_ascii=False)
        return {'csv': csv_path, 'summary_path': summary_path, 'summary': resumo}

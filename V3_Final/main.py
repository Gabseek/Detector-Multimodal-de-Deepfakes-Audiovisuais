# Configura o programa e coordena os modos de análise.
import os
import json
import time

DEFAULT_PERFORMANCE_MODE = 'BALANCEADO'
PERFORMANCE_MODE = DEFAULT_PERFORMANCE_MODE
DEFAULT_ANALYSIS_MODE = 'MULTIMODAL'
ANALYSIS_MODE = DEFAULT_ANALYSIS_MODE
ANALYSIS_MODE_LABELS = {'VISUAL': 'VISUAL', 'AUDIO': 'SONORA', 'MULTIMODAL': 'MULTIMODAL COMPLETA'}
ANALYSIS_MODE_OUTPUT_SLUGS = {'VISUAL': 'visual', 'AUDIO': 'audio', 'MULTIMODAL': 'multimodal'}

from pipeline.execution import PerformanceMonitor, ResourceManager, save_dataset_performance_csv

RESOURCE_MANAGER = ResourceManager(PERFORMANCE_MODE)
RESOURCE_MANAGER.apply_environment_limits()

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pipeline.preprocessing import VideoPreprocessor, FrameExtractor
from pipeline.visual import VisualDeepfakeDetector, DetectorMarcasDaguaVisiveis, DetectorForenseSPSL, avaliar_atalho_spsl, como_resultado_visual_spsl
from pipeline.audio import ThreadAwareDFArenaAudioDetector
from pipeline.fusion import MultimodalFusionAnalyzer
from pipeline.execution import GenericDatasetProcessor

RESOURCE_MANAGER.apply_runtime_limits()
EFFORT_FACE_WEIGHTS = 'models/visual/effort_clip_L14_trainOn_FaceForensic.pth'
EFFORT_GENIMAGE_WEIGHTS = 'models/visual/effort_clip_L14_trainOn_sdv14.pth'
EFFORT_WEIGHTS = str(Path(EFFORT_FACE_WEIGHTS) if Path(EFFORT_FACE_WEIGHTS).is_file() else Path(EFFORT_GENIMAGE_WEIGHTS))
EFFORT_CLIP_MODEL = 'openai/clip-vit-large-patch14'
HF_CACHE_DIR = 'models/visual/huggingface'
YUNET_FACE_MODEL = 'models/visual/face_detection_yunet_2023mar.onnx'
DF_ARENA_AUDIO_MODEL = 'models/audio/df_arena_1b_int8.onnx'
INPUT_DIR = Path('input')
DATASET_PADRAO = 'input'
FRAME_INTERVAL = 0.5
WATERMARK_SCREENING_ENABLED = True
WATERMARK_MAX_FRAMES = 5
WATERMARK_MIN_OCR_CONFIDENCE = 0.78
WATERMARK_MAX_WIDTH = 1440
SPSL_MODO = 'CASCATA_EXPERIMENTAL'
SPSL_DIAGNOSTICO_ENABLED = True
SPSL_LIMIAR_ATALHO = 0.98
SPSL_LIMIAR_FRAME_ATALHO = 0.90
SPSL_WEIGHTS = 'models/visual/spsl_best.pth'
SPSL_MAX_FRAMES = 8
SPSL_DEVICE = 'cpu'
EFFORT_VISUAL_NOT_SUSPICIOUS_MAX = 0.35
EFFORT_VISUAL_SUSPICIOUS_MIN = 0.65
EFFORT_VISUAL_MIN_SUSPICIOUS_CONSISTENCY = 0.65
EFFORT_VISUAL_MAX_FRAMES = 8
EFFORT_GUIDED_SCAN_MAX_FRAMES = 60
EFFORT_GUIDED_MAX_FRAMES = 12
EFFORT_GUIDED_MIN_FACE_SIDE = 40
EFFORT_GUIDED_FACE_MARGIN = 0.30
YUNET_SCORE_THRESHOLD = 0.80
YUNET_NMS_THRESHOLD = 0.30
YUNET_INPUT_MAX_SIDE = 640
SCENE_BHATTACHARYYA_MIN = 0.55
SCENE_MAD_MIN = 0.12
DF_ARENA_THRESHOLD = 0.50
AUDIO_SUSPICIOUS_FRACTION = 0.25
FUSION_WEIGHT_VISUAL = 0.50
FUSION_WEIGHT_AUDIO = 0.50
FUSION_SUSPICIOUS_THRESHOLD = 0.67
FUSION_NOT_SUSPICIOUS_THRESHOLD = 0.47
FUSION_MAX_SUSPICIOUS_FRAMES = 5
FUSION_MAX_SUSPICIOUS_TIMESTAMPS = 12


# Os nomes dos modos são padronizados antes de iniciar a análise.
def normalizar_modo_analise(modo=None):
    aliases = {
        'VISUAL': 'VISUAL',
        'VIDEO': 'VISUAL',
        'AUDIO': 'AUDIO',
        'SONORA': 'AUDIO',
        'SOM': 'AUDIO',
        'MULTIMODAL': 'MULTIMODAL',
        'COMPLETO': 'MULTIMODAL',
        'CONJUNTO': 'MULTIMODAL'
    }
    chave = str(modo or ANALYSIS_MODE or DEFAULT_ANALYSIS_MODE).strip().upper()
    if chave not in aliases:
        raise ValueError(f'Modo de análise inválido: {modo}')
    return aliases[chave]


def nome_modo_analise(modo=None):
    modo = normalizar_modo_analise(modo)
    return ANALYSIS_MODE_LABELS[modo]


def slug_modo_analise(modo=None):
    modo = normalizar_modo_analise(modo)
    return ANALYSIS_MODE_OUTPUT_SLUGS[modo]


def modo_inclui_visual(modo=None):
    return normalizar_modo_analise(modo) in {'VISUAL', 'MULTIMODAL'}


def modo_inclui_audio(modo=None):
    return normalizar_modo_analise(modo) in {'AUDIO', 'MULTIMODAL'}


def modo_requer_audio(modo=None):
    return normalizar_modo_analise(modo) in {'AUDIO', 'MULTIMODAL'}


def format_score(valor):
    if valor is None:
        return 'N/A'
    return f'{valor:.4f}'


def format_percentual_opcional(valor):
    try:
        percentual = float(valor)
    except (TypeError, ValueError):
        return 'N/A'
    if not 0.0 <= percentual <= 1.0:
        return 'N/A'
    return f'{percentual:.2%}'


def formatar_duracao(segundos):
    if segundos is None:
        return 'N/A'
    segundos = max(0.0, float(segundos))
    horas = int(segundos // 3600)
    restante = segundos - horas * 3600
    minutos = int(restante // 60)
    segundos_restantes = restante - minutos * 60
    return f'{horas:02d}h {minutos:02d}m {segundos_restantes:05.2f}s'


def classificar_risco_visual(visual):
    classificacao = str((visual or {}).get('classification', '')).strip().upper()
    if classificacao == 'SUSPEITO':
        return 'SUSPEITO'
    if classificacao in {'AUTENTICO', 'NAO_SUSPEITO'}:
        return 'NAO_SUSPEITO'
    score = (visual or {}).get('visual_score')
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 'SUSPEITO'
    return 'SUSPEITO' if score >= (EFFORT_VISUAL_NOT_SUSPICIOUS_MAX + EFFORT_VISUAL_SUSPICIOUS_MIN) / 2.0 else 'NAO_SUSPEITO'


def classificar_risco_audio(audio):
    classificacao = str((audio or {}).get('classification', '')).strip().upper()
    if classificacao == 'SINTETICO':
        return 'SUSPEITO'
    if classificacao == 'AUTENTICO':
        return 'NAO_SUSPEITO'
    score = (audio or {}).get('spoof_score')
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 'SUSPEITO'
    return 'SUSPEITO' if score >= DF_ARENA_THRESHOLD else 'NAO_SUSPEITO'


def verify_visual_models():
    if not Path(EFFORT_WEIGHTS).exists():
        raise FileNotFoundError(f'Nenhum checkpoint Effort visual foi encontrado. Coloque o checkpoint facial oficial em {EFFORT_FACE_WEIGHTS} ou mantenha o checkpoint GenImage em {EFFORT_WEIGHTS}. Consulte o README do Effort-AIGI-Detection para obter os pesos.')
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def verify_audio_model():
    if not Path(DF_ARENA_AUDIO_MODEL).exists():
        raise FileNotFoundError(f'Modelo DF-Arena-1B não encontrado: {DF_ARENA_AUDIO_MODEL}')


def confirm_visual_only(video_path):
    print('\n' + '=' * 70)
    print('AVISO - VÍDEO SEM ÁUDIO')
    print('=' * 70)
    print(f'Vídeo: {video_path}')
    print('Sem áudio não será possível executar o detector sonoro.')
    while True:
        resposta = input('Continuar somente com a análise visual? [s/n]: ').strip().lower()
        if resposta in {'s', 'sim'}:
            return True
        if resposta in {'n', 'nao', 'não'}:
            return False
        print('Resposta inválida.')


def create_dataset_audio_confirmation():
    estado = {'policy': None}

    def confirmar(video_path):
        if estado['policy'] == 'continue_all':
            return True
        if estado['policy'] == 'skip_all':
            return False
        print('\n' + '=' * 70)
        print('VÍDEO DA PASTA SEM ÁUDIO')
        print('=' * 70)
        print(video_path)
        print('[s] Processar somente visualmente')
        print('[n] Pular este vídeo')
        print('[t] Processar visualmente todos os próximos sem áudio')
        print('[p] Pular todos os próximos sem áudio')
        while True:
            resposta = input('Escolha: ').strip().lower()
            if resposta == 's':
                return True
            if resposta == 'n':
                return False
            if resposta == 't':
                estado['policy'] = 'continue_all'
                return True
            if resposta == 'p':
                estado['policy'] = 'skip_all'
                return False
            print('Opção inválida.')
    return confirmar


# Os pesos e limiares definidos acima são passados à fusão multimodal.
def criar_fusor_multimodal():
    return MultimodalFusionAnalyzer(
        peso_visual=FUSION_WEIGHT_VISUAL,
        peso_audio=FUSION_WEIGHT_AUDIO,
        limite_suspeito=FUSION_SUSPICIOUS_THRESHOLD,
        limite_nao_suspeito=FUSION_NOT_SUSPICIOUS_THRESHOLD,
        max_frames_suspeitos=FUSION_MAX_SUSPICIOUS_FRAMES,
        max_timestamps_suspeitos=FUSION_MAX_SUSPICIOUS_TIMESTAMPS
    )


# Prepara os dados do vídeo para os detectores.
def executar_preprocessamento(video_path, output_dir, monitor, extrair_audio=True):
    preprocessor = VideoPreprocessor(output_dir=output_dir)
    print('\n' + '=' * 70)
    print('ETAPA 1 - PRÉ-PROCESSAMENTO')
    print('=' * 70)
    print('\nObtendo metadados com FFprobe...')
    with monitor.measure('pre.ffprobe', 'FFprobe - leitura de metadados', 'PRE_PROCESSAMENTO'):
        video_info = preprocessor.get_video_info(video_path)
    has_video = preprocessor.has_video_stream(video_info)
    has_audio = preprocessor.has_audio_stream(video_info)
    if not has_video:
        raise RuntimeError('O arquivo informado não possui uma stream de vídeo válida.')
    audio_path = None
    if has_audio and extrair_audio:
        print('Stream de áudio encontrada.')
        print('Extraindo áudio em WAV mono 16 kHz...')
        with monitor.measure(
            'pre.ffmpeg_audio',
            'FFmpeg - extração/normalização do áudio',
            'PRE_PROCESSAMENTO'
        ):
            audio_path = preprocessor.extract_audio(video_path)
        print(f'Áudio extraído: {audio_path}')
    elif has_audio:
        print('Stream de áudio encontrada, mas a modalidade atual não precisa de áudio. A extração WAV será ignorada.')
    else:
        print('\nNenhuma stream de áudio foi encontrada no vídeo.')
    return {
        'original_video': Path(video_path),
        'audio': audio_path,
        'has_audio': has_audio,
        'has_video': has_video,
        'video_info': video_info
    }


# Executa a cascata visual de acordo com as opções escolhidas.
def executar_ramo_visual(preprocessing_data, output_dir, detector_cache, monitor):
    with monitor.measure(
        'branch.visual',
        'Ramo visual completo',
        'VISUAL',
        count_in_sequential_sum=False
    ):
        print('\n' + '=' * 70)
        print('RAMO VISUAL - EXTRAÇÃO DE FRAMES')
        print('=' * 70)
        marca_resultado = None
        if WATERMARK_SCREENING_ENABLED:
            try:
                if 'watermark' not in detector_cache:
                    with monitor.measure(
                        'visual.watermark_load',
                        'RapidOCR - carregamento',
                        'VISUAL'
                    ):
                        detector_cache['watermark'] = DetectorMarcasDaguaVisiveis(
                            quadros_maximos=WATERMARK_MAX_FRAMES,
                            largura_maxima=WATERMARK_MAX_WIDTH,
                            confianca_ocr_minima=WATERMARK_MIN_OCR_CONFIDENCE
                        )
                else:
                    monitor.record_cached(
                        'visual.watermark_load',
                        'RapidOCR - modelo ja carregado',
                        'VISUAL'
                    )
                with monitor.measure(
                    'visual.watermark_scan',
                    'RapidOCR - marcas visiveis (inclui espelhamento)',
                    'VISUAL'
                ):
                    marca_resultado = detector_cache['watermark'].analisar(
                        caminho_video=preprocessing_data['original_video'],
                        saida_json=output_dir / 'visual_watermark.json'
                    )
                print(f"Triagem de marcas: {marca_resultado['status']} | marca={marca_resultado.get('marca_detectada') or 'N/A'} | frames={marca_resultado.get('frames_amostrados', 0)}")
                if marca_resultado['status'] == 'CONFIRMADA':
                    visual_data = DetectorMarcasDaguaVisiveis.como_resultado_visual(
                        marca_resultado,
                        output_dir / 'visual_score.json'
                    )
                    visual_data['spsl_diagnostico'] = {
                        'status': 'IGNORADO_POR_MARCA',
                        'score_spsl': None,
                        'modo': SPSL_MODO,
                        'interrompe_effort': False
                    }
                    (output_dir / 'visual_score.json').write_text(
                        json.dumps(visual_data, ensure_ascii=False, indent=2, default=str),
                        encoding='utf-8'
                    )
                    print('SUSPEITO: marca visivel confirmada: ' + str(marca_resultado['marca_detectada']) + '. Effort dispensado; demais modalidades preservadas.')
                    print('Quadros de confirmação: ' + str(marca_resultado.get('pasta_frames_evidencia') or 'não salvos'))
                    print('Imagens anotadas: ' + str(marca_resultado.get('pasta_imagens_evidencia') or 'não salvas'))
                    for aviso in marca_resultado.get('avisos_salvamento', []):
                        print("AVISO - gravação da marca d'água: " + str(aviso))
                    return {
                        'frame_data': {
                            'frames': marca_resultado.get('frames_evidencia', []),
                            'origem': 'ATALHO_MARCA_DAGUA',
                            'fps': marca_resultado.get('fps'),
                            'duration': marca_resultado.get('duration'),
                            'total_frames': marca_resultado.get('total_frames')
                        },
                        'face_data': {
                            'faces': [],
                            'total_faces': 0,
                            'detector': 'NAO_EXECUTADO_MARCA_DAGUA'
                        },
                        'visual_data': visual_data
                    }
            except Exception as error:
                print(f'AVISO: triagem OCR indisponivel/falhou: {type(error).__name__}: {error}. O Effort sera executado normalmente.')
                marca_resultado = {
                    'status': 'ERRO',
                    'erro': f'{type(error).__name__}: {error}',
                    'marca_detectada': None
                }
                (output_dir / 'visual_watermark.json').write_text(
                    json.dumps(marca_resultado, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
        spsl_resultado = {
            'status': 'DESATIVADO',
            'score_spsl': None,
            'modo': SPSL_MODO,
            'interrompe_effort': False
        }
        if SPSL_DIAGNOSTICO_ENABLED:
            try:
                if 'spsl' not in detector_cache:
                    with monitor.measure(
                        'visual.spsl_load',
                        'SPSL - inicialização do detector',
                        'VISUAL'
                    ):
                        detector_cache['spsl'] = DetectorForenseSPSL(
                            pesos=SPSL_WEIGHTS,
                            yunet=YUNET_FACE_MODEL,
                            max_frames=SPSL_MAX_FRAMES,
                            dispositivo=SPSL_DEVICE,
                            cpu_threads=RESOURCE_MANAGER.cpu_threads_per_heavy_task
                        )
                else:
                    monitor.record_cached('visual.spsl_load', 'SPSL - detector em cache', 'VISUAL')
                with monitor.measure(
                    'visual.spsl_inference',
                    'SPSL - diagnóstico facial',
                    'VISUAL'
                ):
                    spsl_resultado = detector_cache['spsl'].analisar(
                        video=preprocessing_data['original_video'],
                        destino=output_dir / 'spsl_diagnostico.json'
                    )
                print(f"SPSL ({SPSL_MODO}): {spsl_resultado.get('status')} | score={spsl_resultado.get('score_spsl')} | frames={spsl_resultado.get('frames_analisados', 0)}")
            except Exception as erro:
                print(f'AVISO: SPSL indisponível ({erro}); mantendo Effort.')
                spsl_resultado = {
                    'status': 'INDISPONIVEL',
                    'score_spsl': None,
                    'erro': f'{type(erro).__name__}: {erro}',
                    'modo': SPSL_MODO,
                    'interrompe_effort': False
                }
        spsl_resultado['modo'] = SPSL_MODO
        criterio_spsl = avaliar_atalho_spsl(
            spsl_resultado,
            modo=SPSL_MODO,
            limiar_medio=SPSL_LIMIAR_ATALHO,
            limiar_frame=SPSL_LIMIAR_FRAME_ATALHO
        )
        spsl_resultado['criterios_atalho'] = criterio_spsl
        spsl_resultado['interrompe_effort'] = bool(criterio_spsl['elegivel'])
        if SPSL_DIAGNOSTICO_ENABLED:
            (output_dir / 'spsl_diagnostico.json').write_text(
                json.dumps(spsl_resultado, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
        print(f"SPSL gate: {('ATALHO AUTORIZADO' if criterio_spsl['elegivel'] else 'SEGUIR EFFORT')} | quadros fortes={criterio_spsl['frames_fortes']}/{criterio_spsl['frames_com_rosto_qualificado']} | motivos={','.join(criterio_spsl['motivos_nao_atalho']) or 'EVIDENCIA_CONSISTENTE'}")
        if criterio_spsl['elegivel']:
            visual_data = como_resultado_visual_spsl(spsl_resultado, criterio_spsl)
            visual_data['status_triagem_marca'] = marca_resultado.get('status') if marca_resultado else 'DESATIVADA'
            (output_dir / 'visual_score.json').write_text(
                json.dumps(visual_data, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
            print('SPSL: evidência persistente, classificação visual SUSPEITO. Effort dispensado. A análise sonora continua disponível.')
            return {
                'frame_data': {'frames': [], 'origem': 'ATALHO_SPSL'},
                'face_data': {
                    'faces': [],
                    'total_faces': visual_data['faces_analyzed'],
                    'detector': 'YUNET_INTERNO_SPSL'
                },
                'visual_data': visual_data
            }
        frame_extractor = FrameExtractor(
            output_dir=output_dir / 'frames',
            interval_seconds=FRAME_INTERVAL
        )
        with monitor.measure('visual.frame_extraction', 'OpenCV - extração de frames', 'VISUAL'):
            frame_data = frame_extractor.process(preprocessing_data['original_video'])
        print('\n' + '=' * 70)
        print('RAMO VISUAL - EFFORT DUAL VIEW (SCENE/FACE-GUIDED, SEM FUSAO)')
        print('=' * 70)
        if 'visual' not in detector_cache:
            with monitor.measure(
                'visual.detectors_load',
                'Effort + YuNet - carregamento',
                'VISUAL'
            ):
                detector_cache['visual'] = VisualDeepfakeDetector(
                    effort_weights_path=EFFORT_WEIGHTS,
                    effort_clip_model_name=EFFORT_CLIP_MODEL,
                    hf_cache_dir=HF_CACHE_DIR,
                    output_path=output_dir / 'visual_score.json',
                    limite_nao_suspeito=EFFORT_VISUAL_NOT_SUSPICIOUS_MAX,
                    limite_suspeito=EFFORT_VISUAL_SUSPICIOUS_MIN,
                    consistencia_minima_suspeito=EFFORT_VISUAL_MIN_SUSPICIOUS_CONSISTENCY,
                    max_frames=EFFORT_VISUAL_MAX_FRAMES,
                    yunet_model_path=YUNET_FACE_MODEL,
                    yunet_score_threshold=YUNET_SCORE_THRESHOLD,
                    yunet_nms_threshold=YUNET_NMS_THRESHOLD,
                    yunet_input_max_side=YUNET_INPUT_MAX_SIDE,
                    guided_scan_max_frames=EFFORT_GUIDED_SCAN_MAX_FRAMES,
                    guided_max_frames=EFFORT_GUIDED_MAX_FRAMES,
                    guided_min_face_side=EFFORT_GUIDED_MIN_FACE_SIDE,
                    guided_face_margin=EFFORT_GUIDED_FACE_MARGIN,
                    scene_bhattacharyya_min=SCENE_BHATTACHARYYA_MIN,
                    scene_mad_min=SCENE_MAD_MIN
                )
        else:
            monitor.record_cached(
                'visual.detectors_load',
                'Effort + YuNet - modelos já carregados',
                'VISUAL'
            )
        visual_detector = detector_cache['visual']
        visual_detector.output_path = output_dir / 'visual_score.json'
        with monitor.measure(
            'visual.detectors_inference',
            'Effort dual-view - screening scene/face-guided separado',
            'VISUAL'
        ):
            visual_data = visual_detector.process(
                frame_data=frame_data,
                video_path=preprocessing_data['original_video']
            )
        visual_data['spsl_diagnostico'] = spsl_resultado
        visual_data['spsl_executado'] = spsl_resultado.get('status') == 'OK'
        visual_data['spsl_interrompeu_effort'] = False
        visual_data['criterios_atalho_spsl'] = criterio_spsl
        visual_data['marca_dagua_confirmada'] = False
        visual_data['marca_dagua_detectada'] = None
        visual_data['etapa_visual_executada'] = 'EFFORT_APOS_TRIAGEM'
        visual_data['effort_executado'] = True
        visual_data['status_triagem_marca'] = marca_resultado.get('status') if marca_resultado else 'DESATIVADA'
        (output_dir / 'visual_score.json').write_text(
            json.dumps(visual_data, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )
        face_data = {
            'faces': [],
            'total_faces': int(visual_data.get('faces_analyzed', 0) or 0),
            'detector': 'YUNET_INTERNO_EFFORT_DUALVIEW'
        }
        return {'frame_data': frame_data, 'face_data': face_data, 'visual_data': visual_data}


# Analisa a faixa sonora de forma independente.
def executar_ramo_audio(audio_path, output_dir, detector_cache, monitor):
    with monitor.measure(
        'branch.audio',
        'Ramo de detecção de áudio',
        'AUDIO',
        count_in_sequential_sum=False
    ):
        print('\n' + '=' * 70)
        print('RAMO DE ÁUDIO - DF-ARENA-1B')
        print('=' * 70)
        verify_audio_model()
        if 'audio' not in detector_cache:
            with monitor.measure(
                'audio.df_arena_load',
                'DF-Arena-1B - carregamento ONNX INT8',
                'AUDIO'
            ):
                detector_cache['audio'] = ThreadAwareDFArenaAudioDetector(
                    model_path=DF_ARENA_AUDIO_MODEL,
                    output_path=output_dir / 'audio_score.json',
                    threshold=DF_ARENA_THRESHOLD,
                    suspicious_fraction=AUDIO_SUSPICIOUS_FRACTION,
                    cpu_threads=RESOURCE_MANAGER.cpu_threads_per_heavy_task
                )
        else:
            monitor.record_cached(
                'audio.df_arena_load',
                'DF-Arena-1B - modelo já carregado',
                'AUDIO'
            )
        detector = detector_cache['audio']
        detector.output_path = output_dir / 'audio_score.json'
        with monitor.measure(
            'audio.df_arena_inference',
            'DF-Arena-1B - análise do áudio completo',
            'AUDIO'
        ):
            audio_data = detector.process(audio_path)
        return audio_data


# Organiza os ramos que podem ser executados ao mesmo tempo.
def executar_ramos(preprocessing_data, output_dir, detector_cache, monitor, analysis_mode):
    modo = normalizar_modo_analise(analysis_mode)
    tarefas = []
    if modo_inclui_visual(modo):
        tarefas.append((
            'visual',
            executar_ramo_visual,
            (preprocessing_data, output_dir, detector_cache, monitor)
        ))
    if modo_inclui_audio(modo) and preprocessing_data['has_audio']:
        tarefas.append((
            'audio',
            executar_ramo_audio,
            (preprocessing_data['audio'], output_dir, detector_cache, monitor)
        ))
    resultados = {'visual': None, 'audio': None}
    if RESOURCE_MANAGER.heavy_workers <= 1 or len(tarefas) <= 1:
        print('\nExecução dos ramos: SEQUENCIAL')
        for nome, funcao, args in tarefas:
            resultados[nome] = funcao(*args)
    else:
        print(f'\nExecução dos ramos: PARALELA CONTROLADA ({min(
            RESOURCE_MANAGER.heavy_workers,
            len(tarefas)
        )} ramos)')
        with ThreadPoolExecutor(
            max_workers=min(RESOURCE_MANAGER.heavy_workers, len(tarefas)),
            thread_name_prefix='pipeline'
        ) as executor:
            futuros = {executor.submit(funcao, *args): nome for nome, funcao, args in tarefas}
            for futuro in as_completed(futuros):
                nome = futuros[futuro]
                resultados[nome] = futuro.result()
                print(f'[Ramo concluído] {nome.upper()}')
    return resultados


# Reúne os resultados dos ramos e faz a fusão quando necessário.
def run_pipeline(
    video_path,
    output_dir=None,
    audio_confirmation=None,
    detector_cache=None,
    analysis_mode=None
):
    video_path = Path(video_path)
    requested_mode = normalizar_modo_analise(analysis_mode or ANALYSIS_MODE)
    if modo_inclui_visual(requested_mode):
        verify_visual_models()
    if not video_path.exists():
        raise FileNotFoundError(f'Vídeo não encontrado: {video_path}')
    detector_cache = detector_cache if detector_cache is not None else {}
    output_dir = Path(output_dir) if output_dir else Path('output') / video_path.stem
    if requested_mode != 'MULTIMODAL' and output_dir == Path('output') / video_path.stem:
        output_dir = output_dir / ('modo_' + slug_modo_analise(requested_mode))
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = PerformanceMonitor(resource_config=RESOURCE_MANAGER.as_dict())
    monitor.start_application()
    print(f'\nMODO DE ANÁLISE: {nome_modo_analise(requested_mode)}')
    RESOURCE_MANAGER.print_summary()
    pre = executar_preprocessamento(
        video_path,
        output_dir,
        monitor,
        extrair_audio=modo_requer_audio(requested_mode)
    )
    effective_mode = requested_mode
    if modo_requer_audio(requested_mode) and (not pre['has_audio']):
        if requested_mode == 'MULTIMODAL':
            autorizado = confirm_visual_only(video_path) if audio_confirmation is None else audio_confirmation(video_path)
            if autorizado:
                effective_mode = 'VISUAL'
            else:
                monitor.finish_application()
                performance = monitor.save(output_dir / 'performance')
                return dict(
                    status='IGNORADO_SEM_AUDIO',
                    video=video_path,
                    output_dir=output_dir,
                    has_audio=False,
                    visual_only=True,
                    analysis_mode_requested=requested_mode,
                    analysis_mode_effective=None,
                    visual=None,
                    audio=None,
                    fusion=None,
                    performance=performance
                )
        else:
            print('Vídeo sem áudio: análise sonora ignorada.')
            monitor.finish_application()
            performance = monitor.save(output_dir / 'performance')
            return dict(
                status='IGNORADO_SEM_AUDIO',
                video=video_path,
                output_dir=output_dir,
                has_audio=False,
                visual_only=False,
                analysis_mode_requested=requested_mode,
                analysis_mode_effective=None,
                visual=None,
                audio=None,
                fusion=None,
                performance=performance
            )
    branches = executar_ramos(pre, output_dir, detector_cache, monitor, effective_mode)
    visual_branch = branches.get('visual') or {}
    frame_data = visual_branch.get('frame_data')
    face_data = visual_branch.get('face_data')
    visual_data = visual_branch.get('visual_data')
    audio_data = branches.get('audio')
    fusion_data = None
    if requested_mode == 'MULTIMODAL':
        if 'fusion' not in detector_cache:
            detector_cache['fusion'] = criar_fusor_multimodal()
        with monitor.measure('fusion.multimodal', 'Fusão multimodal + relatório', 'FUSAO'):
            fusion_data = detector_cache['fusion'].process(
                visual_data=visual_data,
                audio_data=audio_data,
                frame_data=frame_data,
                face_data=face_data,
                output_dir=output_dir / 'fusion'
            )
    monitor.finish_application()
    performance = monitor.save(output_dir / 'performance')
    monitor.print_report()
    return dict(
        status='OK_VISUAL_ONLY' if requested_mode == 'MULTIMODAL' and effective_mode == 'VISUAL' else 'OK',
        video=video_path,
        output_dir=output_dir,
        has_audio=pre['has_audio'],
        visual_only=effective_mode == 'VISUAL',
        analysis_mode_requested=requested_mode,
        analysis_mode_effective=effective_mode,
        audio_path=pre.get('audio'),
        video_info=pre.get('video_info'),
        frame_data=frame_data,
        face_data=face_data,
        visual=visual_data,
        audio=audio_data,
        fusion=fusion_data,
        performance=performance
    )


def show_result(result):
    if result.get('status') == 'IGNORADO_SEM_AUDIO':
        print('\nVídeo ignorado por ausência de áudio.')
        return
    print('\n' + '=' * 72 + '\nRESULTADO FINAL DO VÍDEO\n' + '=' * 72)
    print(f"Vídeo: {result.get('video')}")
    visual, audio, fusion = (result.get(x) or {} for x in ('visual', 'audio', 'fusion'))
    if visual:
        print(f"VISUAL: {visual.get('classification')} | score={format_score(visual.get('visual_score'))}")
        print(f"Etapa visual: {visual.get('etapa_visual_executada')}")
    if audio:
        print(f"SONORA: {audio.get('classification')} | score={format_score(audio.get('spoof_score'))}")
    if fusion:
        print(f"FUSÃO MULTIMODAL: {fusion.get('classification')} | score={format_score(fusion.get('suspicion_score'))}")
        print(f"Interpretação: {fusion.get('interpretation')}")
    perf = result.get('performance') or {}
    print(f"TEMPO TOTAL: {perf.get('application_wall_time_seconds', 0):.3f} s")


def criar_registro_dataset(indice, video_path, relative_path, video_output, resultado):
    visual, audio, fusion = (resultado.get(x) or {} for x in ('visual', 'audio', 'fusion'))
    segmento = audio.get('most_suspicious_segment') or audio.get('most_suspicious_window') or {}
    frames = fusion.get('suspicious_frames') or []
    timestamps = fusion.get('suspicious_timestamps') or []
    registro = {
        'indice': indice,
        'video': video_path.name,
        'caminho_relativo': str(relative_path),
        'caminho_completo': str(video_path.resolve()),
        'possui_audio': resultado.get('has_audio'),
        'modo': resultado.get('analysis_mode_effective') or resultado.get('analysis_mode_requested'),
        'marca_dagua_confirmada': visual.get('marca_dagua_confirmada', False),
        'marca_dagua_detectada': visual.get('marca_dagua_detectada'),
        'etapa_visual_executada': visual.get('etapa_visual_executada'),
        'effort_executado': visual.get('effort_executado') if visual else None,
        'status_triagem_marca': visual.get('status_triagem_marca'),
        'modo_spsl': (visual.get('spsl_diagnostico') or {}).get('modo', SPSL_MODO),
        'spsl_interrompeu_effort': visual.get('spsl_interrompeu_effort', False),
        'spsl_atalho_elegivel': (visual.get('criterios_atalho_spsl') or {}).get('elegivel'),
        'spsl_frames_fortes': (visual.get('criterios_atalho_spsl') or {}).get('frames_fortes'),
        'status_spsl': (visual.get('spsl_diagnostico') or {}).get('status'),
        'score_spsl': (visual.get('spsl_diagnostico') or {}).get('score_spsl'),
        'score_medio_spsl': (visual.get('spsl_diagnostico') or {}).get('score_medio'),
        'score_top25_spsl': (visual.get('spsl_diagnostico') or {}).get('score_top25'),
        'classificacao_spsl_exploratoria': (visual.get('spsl_diagnostico') or {}).get('classificacao_exploratoria'),
        'frames_spsl': (visual.get('spsl_diagnostico') or {}).get('frames_analisados'),
        'tempo_spsl_segundos': (visual.get('spsl_diagnostico') or {}).get('tempo_segundos'),
        'tempo_total_video_segundos': (resultado.get('performance') or {}).get('application_wall_time_seconds'),
        'score_visual': visual.get('visual_score'),
        'classificacao_visual': visual.get('classification'),
        'classificacao_risco_visual': classificar_risco_visual(visual) if visual else None,
        'frames_analisados': visual.get('frames_analyzed'),
        'rostos_analisados': visual.get('faces_analyzed'),
        'score_effort_guided_full': visual.get('score_effort_guided_full'),
        'classificacao_effort_guided_full': visual.get('classification_effort_guided_full'),
        'score_effort_guided_face': visual.get('score_effort_guided_face'),
        'classificacao_effort_guided_face': visual.get('classification_effort_guided_face'),
        'faces_guiadas_selecionadas': (visual.get('guided_selection') or {}).get('selected_face_frames'),
        'taxa_deteccao_face_scan': (visual.get('guided_selection') or {}).get('face_detection_rate'),
        'score_audio': audio.get('spoof_score'),
        'bonafide_score': audio.get('bonafide_score'),
        'classificacao_audio': audio.get('classification'),
        'classificacao_risco_audio': classificar_risco_audio(audio) if audio else None,
        'janelas_audio_analisadas': audio.get('segments_analyzed'),
        'trecho_audio_suspeito_inicio': segmento.get('start_time'),
        'trecho_audio_suspeito_fim': segmento.get('end_time'),
        'trecho_audio_spoof_score': segmento.get('spoof_score'),
        'classificacao_multimodal': fusion.get('classification'),
        'score_suspeita_multimodal': fusion.get('suspicion_score'),
        'confianca_multimodal': fusion.get('confidence'),
        'modalidades_usadas_fusao': '; '.join(map(str, fusion.get('modalities_used') or [])),
        'quantidade_modalidades_fusao': len(fusion.get('modalities_used') or []),
        'frames_suspeitos_quantidade': len(frames),
        'frames_suspeitos_resumo': json.dumps(frames, ensure_ascii=False, default=str),
        'timestamps_suspeitos_quantidade': len(timestamps),
        'timestamps_suspeitos_resumo': json.dumps(timestamps, ensure_ascii=False, default=str),
        'arquivo_relatorio_multimodal_json': fusion.get('json_path'),
        'arquivo_relatorio_multimodal_txt': fusion.get('txt_path'),
        'status': resultado.get('status'),
        'diretorio_saida': str(video_output)
    }
    return registro


# Permite escolher se a busca por marcas visíveis será usada.
def configurar_rapidocr():
    global WATERMARK_SCREENING_ENABLED
    if ANALYSIS_MODE != 'MULTIMODAL':
        WATERMARK_SCREENING_ENABLED = True
        return
    while True:
        resposta = input('Utilizar RapidOCR na fusão multimodal? [s/n]: ').strip().lower()
        if resposta in {'s', 'sim'}:
            WATERMARK_SCREENING_ENABLED = True
            break
        if resposta in {'n', 'nao', 'não'}:
            WATERMARK_SCREENING_ENABLED = False
            break
        print('Responda s ou n.')
    print('RapidOCR: ' + ('ATIVADO' if WATERMARK_SCREENING_ENABLED else 'DESATIVADO'))


def assinatura_video(caminho):
    info = Path(caminho).stat()
    return {'tamanho': info.st_size, 'modificado_ns': info.st_mtime_ns}


def configuracao_execucao():
    return {
        'modalidade': ANALYSIS_MODE,
        'rapidocr': bool(WATERMARK_SCREENING_ENABLED) if modo_inclui_visual() else None,
        'perfil': RESOURCE_MANAGER.mode,
        'spsl_atalho': SPSL_LIMIAR_ATALHO,
        'spsl_frame': SPSL_LIMIAR_FRAME_ATALHO,
        'peso_visual': FUSION_WEIGHT_VISUAL,
        'peso_audio': FUSION_WEIGHT_AUDIO,
        'limiar_fusao': (FUSION_SUSPICIOUS_THRESHOLD + FUSION_NOT_SUSPICIOUS_THRESHOLD) / 2,
        'limiares_fusao': [FUSION_NOT_SUSPICIOUS_THRESHOLD, FUSION_SUSPICIOUS_THRESHOLD],
        'limiar_audio': DF_ARENA_THRESHOLD,
        'fracao_audio': AUDIO_SUSPICIOUS_FRACTION,
        'intervalo_quadros': FRAME_INTERVAL
    }


def salvar_estado_execucao(caminho, estado):
    caminho = Path(caminho)
    temporario = caminho.with_name(caminho.name + '.tmp')
    with temporario.open('w', encoding='utf-8') as arquivo:
        json.dump(estado, arquivo, ensure_ascii=False, default=str)
        arquivo.flush()
        os.fsync(arquivo.fileno())
    os.replace(temporario, caminho)


def resultado_tempos_compacto(resultado):
    perf = resultado.get('performance') or {}
    campos = (
        'application_wall_time_seconds',
        'category_leaf_totals_seconds',
        'sum_of_leaf_task_times_seconds',
        'estimated_parallel_overlap_seconds',
        'measured_parallel_overlap_seconds',
        'measured_active_wall_seconds',
        'unmeasured_wall_seconds',
        'task_time_to_wall_time_ratio',
        'resource_configuration'
    )
    return {
        'video': str(resultado.get('video', '')),
        'status': resultado.get('status', 'OK'),
        'erro': resultado.get('erro', ''),
        'performance': {chave: perf.get(chave) for chave in campos}
    }


def registro_resultado(indice, video_path, relative_path, video_output, resultado, gerenciador):
    registro = criar_registro_dataset(indice, video_path, relative_path, video_output, resultado)
    classe = gerenciador.get_label(relative_path)
    registro['classe_real'] = classe
    for chave, saida in (
        ('correto_visual', 'classificacao_risco_visual'),
        ('correto_audio', 'classificacao_risco_audio'),
        ('correto_multimodal', 'classificacao_multimodal')
    ):
        classificacao = registro.get(saida)
        registro[chave] = classificacao == classe if classe is not None and classificacao is not None else None
    return registro


# Monta o PDF a partir dos registros e do resumo que o programa já salvou.
def gerar_relatorio_pdf(registros, resumo, pasta_saida):
    from datetime import datetime
    from html import escape
    import csv

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable, KeepTogether, LongTable, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle
    )

    pasta_saida = Path(pasta_saida)
    caminho_pdf = pasta_saida / 'relatorio_final.pdf'
    fontes = (
        Path('C:/Windows/Fonts/arial.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    )
    fonte = 'Helvetica'
    for arquivo_fonte in fontes:
        if arquivo_fonte.is_file():
            if 'FonteRelatorio' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('FonteRelatorio', str(arquivo_fonte)))
            fonte = 'FonteRelatorio'
            break

    azul = colors.HexColor('#19364B')
    destaque = colors.HexColor('#226F8A')
    cinza = colors.HexColor('#576775')
    fundo = colors.HexColor('#F2F6F8')
    borda = colors.HexColor('#D7E1E8')
    claro = colors.white
    titulo = ParagraphStyle('TituloPDF', fontName=fonte, fontSize=19, leading=24, textColor=azul, spaceAfter=8)
    subtitulo = ParagraphStyle('SubtituloPDF', fontName=fonte, fontSize=12, leading=16, textColor=azul, spaceBefore=15, spaceAfter=8)
    texto = ParagraphStyle('TextoPDF', fontName=fonte, fontSize=9, leading=14, textColor=azul, spaceAfter=7)
    pequeno = ParagraphStyle('PequenoPDF', fontName=fonte, fontSize=7.4, leading=10.5, textColor=cinza)
    celula = ParagraphStyle('CelulaPDF', fontName=fonte, fontSize=7.5, leading=10.5, textColor=azul, splitLongWords=1)
    cabecalho = ParagraphStyle('CabecalhoPDF', fontName=fonte, fontSize=8, leading=11, textColor=claro)
    numero = ParagraphStyle('NumeroPDF', fontName=fonte, fontSize=15, leading=19, textColor=azul, alignment=TA_CENTER)
    rotulo = ParagraphStyle('RotuloPDF', fontName=fonte, fontSize=7.5, leading=10, textColor=cinza, alignment=TA_CENTER)
    doc = SimpleDocTemplate(str(caminho_pdf), pagesize=A4,
                            rightMargin=39, leftMargin=39, topMargin=48, bottomMargin=48,
                            title='Relatório final da análise de deepfakes')
    largura = doc.width
    estilo_tabela = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, 0), 0.7, destaque),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [claro, fundo]),
    ]

    def texto_celula(valor):
        return Paragraph(escape(str(valor)) if valor is not None else 'N/D', celula)

    def porcentagem(valor):
        return f'{valor * 100:.2f}%'.replace('.', ',') if isinstance(valor, (int, float)) else 'N/D'

    def segundos(valor):
        try:
            return f'{float(valor):.2f} s'.replace('.', ',')
        except (TypeError, ValueError):
            return 'N/D'

    def desenhar_pagina(canvas, documento):
        canvas.saveState()
        canvas.setStrokeColor(borda)
        canvas.line(39, 39, A4[0] - 39, 39)
        canvas.setFont(fonte, 8)
        canvas.setFillColor(cinza)
        canvas.drawString(39, 27, 'Relatório gerado pelo pipeline de detecção de deepfakes')
        canvas.drawRightString(A4[0] - 39, 27, f'Página {documento.page}')
        canvas.restoreState()

    modo = resumo.get('modo_analise') or 'MULTIMODAL'
    nomes_modo = {'VISUAL': 'Somente visual', 'AUDIO': 'Somente sonora', 'MULTIMODAL': 'Fusão multimodal'}
    campo_saida = {'VISUAL': 'classificacao_risco_visual',
                   'AUDIO': 'classificacao_risco_audio',
                   'MULTIMODAL': 'classificacao_multimodal'}.get(modo, 'classificacao_multimodal')
    campo_score = {'VISUAL': 'score_visual', 'AUDIO': 'score_audio',
                   'MULTIMODAL': 'score_suspeita_multimodal'}.get(modo, 'score_suspeita_multimodal')
    finalizados = [r for r in registros if r.get('status') in {'OK', 'OK_VISUAL_ONLY'}]
    suspeitos = [r for r in finalizados if r.get(campo_saida) == 'SUSPEITO']
    nao_suspeitos = [r for r in finalizados if r.get(campo_saida) == 'NAO_SUSPEITO']
    restantes = [r for r in registros if r not in suspeitos and r not in nao_suspeitos]
    historia = []
    historia.append(Paragraph('RELATÓRIO FINAL DE ANÁLISE', titulo))
    historia.append(Paragraph('Resultados dos vídeos processados e etapas de detecção executadas.', texto))
    historia.append(HRFlowable(width='100%', thickness=1.3, color=destaque, spaceBefore=2, spaceAfter=15))

    dados_cartoes = [
        (str(len(registros)), 'Vídeos encontrados'),
        (str(len(suspeitos)), 'Suspeitos'),
        (str(len(nao_suspeitos)), 'Não suspeitos'),
        (str(len(restantes)), 'Sem classificação')
    ]
    cartoes = Table([[Table([[Paragraph(valor, numero)], [Paragraph(nome, rotulo)]],
                           colWidths=[largura / 4 - 12], hAlign='CENTER')
                      for valor, nome in dados_cartoes]], colWidths=[largura / 4] * 4)
    cartoes.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), fundo),
        ('BOX', (0, 0), (-1, -1), 0.6, borda),
        ('LINEBEFORE', (1, 0), (-1, -1), 0.6, borda),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10)
    ]))
    historia.extend([cartoes, Spacer(1, 12)])
    rapidocr = resumo.get('rapidocr_ativado')
    configuracao = [
        ['Modalidade utilizada', nomes_modo.get(modo, modo)],
        ['Perfil de desempenho', resumo.get('modo_desempenho') or 'N/D'],
        ['RapidOCR', 'Não utilizado' if modo == 'AUDIO' else 'Ativado' if rapidocr else 'Desativado'],
        ['Data do relatório', datetime.now().strftime('%d/%m/%Y %H:%M')]
    ]
    if modo == 'MULTIMODAL':
        limiar = resumo.get('limiar_binario_fusao_multimodal')
        if limiar is not None:
            configuracao.insert(2, ['Limiar de classificação', f'{float(limiar):.2f}'.replace('.', ',')])
    tabela_config = Table([[texto_celula(a), texto_celula(b)] for a, b in configuracao],
                          colWidths=[largura * 0.35, largura * 0.65])
    tabela_config.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                      ('LINEBELOW', (0, 0), (-1, -2), 0.35, borda),
                                      ('TOPPADDING', (0, 0), (-1, -1), 5),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    historia.extend([tabela_config, Spacer(1, 6)])

    historia.append(Paragraph('Tempo de processamento', subtitulo))
    dados_tempos = [
        ['Tempo total acumulado', resumo.get('tempo_total_dataset_formatado', 'N/D')],
        ['Média por vídeo', resumo.get('tempo_medio_video_formatado', 'N/D')],
        ['Menor tempo', resumo.get('tempo_minimo_video_formatado', 'N/D')],
        ['Maior tempo', resumo.get('tempo_maximo_video_formatado', 'N/D')]
    ]
    tabela_tempos = Table([[texto_celula(a), texto_celula(b)] for a, b in dados_tempos],
                          colWidths=[largura * 0.48, largura * 0.52])
    tabela_tempos.setStyle(TableStyle([('ROWBACKGROUNDS', (0, 0), (-1, -1), [fundo, claro]),
                                      ('TOPPADDING', (0, 0), (-1, -1), 5),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    historia.append(tabela_tempos)
    arquivo_tempos = pasta_saida / 'tempos_pipeline.csv'
    if arquivo_tempos.is_file():
        with arquivo_tempos.open(newline='', encoding='utf-8-sig') as arquivo:
            dados_tempos = list(csv.DictReader(arquivo))
        campos_tempos = [
            ('Pré-processamento', 'pre_processamento_seconds'),
            ('Análise visual', 'visual_seconds'),
            ('Análise sonora', 'audio_seconds'),
            ('Fusão multimodal', 'fusao_seconds')
        ]
        medias = []
        for nome, chave in campos_tempos:
            valores = []
            for registro_tempo in dados_tempos:
                try:
                    valor = float(registro_tempo.get(chave) or '')
                    if valor > 0:
                        valores.append(valor)
                except (TypeError, ValueError):
                    pass
            if valores:
                medias.append((nome, segundos(sum(valores) / len(valores))))
        if medias:
            historia.append(Paragraph('Tempo médio por etapa', subtitulo))
            tabela_etapas = Table([[texto_celula(a), texto_celula(b)] for a, b in medias],
                                   colWidths=[largura * 0.48, largura * 0.52])
            tabela_etapas.setStyle(TableStyle([
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [fundo, claro]),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5)
            ]))
            historia.append(tabela_etapas)
            historia.append(Paragraph('Os ramos visual e sonoro podem ser executados em paralelo; '
                                      'seus tempos não devem ser somados para obter o tempo total.', pequeno))
    if resumo.get('videos_reaproveitados_na_retomada'):
        historia.append(Paragraph(
            f"Esta execução reutilizou {resumo['videos_reaproveitados_na_retomada']} vídeo(s) já concluído(s). "
            'O tempo total informado pode reunir sessões diferentes.', pequeno))

    # As métricas classificatórias só aparecem quando há rótulos de referência.
    modalidades = [('Visual', 'metricas_visuais_rotuladas'),
                   ('Sonora', 'metricas_audio_rotuladas'),
                   ('Multimodal', 'metricas_multimodais_rotuladas')]
    metricas_disponiveis = [(nome, resumo.get(chave) or {}) for nome, chave in modalidades
                           if (resumo.get(chave) or {}).get('videos_avaliados', 0)]
    if metricas_disponiveis:
        historia.append(Paragraph('Desempenho com rótulos de referência', subtitulo))
        historia.append(Paragraph('As métricas abaixo usam apenas vídeos com rótulos de referência '
                                  'identificados na entrada. Elas não medem a certeza da '
                                  'classificação de um vídeo isolado.', pequeno))
        campos_metricas = [('Modalidade', 0.17), ('Acur.', 0.12), ('Precisão', 0.12),
                           ('Sensib.', 0.12), ('Espec.', 0.12), ('F1', 0.11),
                           ('Balanc.', 0.12), ('AUC', 0.12)]
        dados_metricas = [[Paragraph(t, cabecalho) for t, _ in campos_metricas]]
        for nome, m in metricas_disponiveis:
            auc = m.get('roc_auc_visual') if nome == 'Visual' else m.get('roc_auc')
            dados_metricas.append([texto_celula(nome)] +
                [texto_celula(porcentagem(m.get(k))) for k in (
                    'acuracia', 'precisao_suspeito', 'sensibilidade_fake',
                    'especificidade_real', 'f1_suspeito', 'balanced_accuracy')] +
                [texto_celula(f'{auc:.4f}'.replace('.', ',') if isinstance(auc, (int, float)) else 'N/D')])
        tabela_metricas = Table(dados_metricas,
                                colWidths=[largura * peso for _, peso in campos_metricas], repeatRows=1)
        tabela_metricas.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), azul)] + estilo_tabela))
        historia.append(tabela_metricas)

        dados_matriz = [[Paragraph(c, cabecalho) for c in ('Modalidade', 'VP', 'VN', 'FP', 'FN', 'Avaliados')]]
        for nome, m in metricas_disponiveis:
            matriz = m.get('matriz_confusao') or {}
            dados_matriz.append([texto_celula(x) for x in (
                nome,
                matriz.get('fake_classificado_suspeito', 'N/D'),
                matriz.get('real_classificado_nao_suspeito', 'N/D'),
                matriz.get('real_classificado_suspeito', 'N/D'),
                matriz.get('fake_classificado_nao_suspeito', 'N/D'),
                m.get('videos_avaliados', 'N/D')
            )])
        tabela_matriz = Table(dados_matriz, colWidths=[largura * 0.25] + [largura * 0.15] * 5,
                              repeatRows=1)
        tabela_matriz.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), azul)] + estilo_tabela))
        historia.append(KeepTogether([
            Paragraph('Matriz de confusão por modalidade', subtitulo),
            Paragraph('Classe positiva: vídeo manipulado (SUSPEITO). '
                      'VP e VN são acertos; FP e FN são erros.', pequeno),
            tabela_matriz
        ]))
    else:
        historia.append(Paragraph('Métricas de avaliação', subtitulo))
        historia.append(Paragraph('Acurácia, precisão, sensibilidade e matriz de confusão não podem '
                                  'ser calculadas sem rótulos reais de referência. O relatório '
                                  'apresenta as classificações obtidas, sem atribuir acertos ou erros.', texto))

    historia.append(Paragraph('Etapas utilizadas', subtitulo))
    historia.append(Paragraph('A coluna "Etapas" na relação abaixo informa os detectores executados '
                              'em cada vídeo. YuNet participa da localização de faces nas etapas '
                              'visuais; SPSL e Effort verificam indícios de manipulação. '
                              'DF-Arena-1B analisa a faixa sonora. RapidOCR procura marcas visíveis, '
                              'quando ativado.', texto))
    historia.append(Paragraph('Os scores são indicadores computacionais de suspeita, não '
                              'probabilidades calibradas nem comprovação definitiva de falsificação.', pequeno))

    # A relação é agrupada pela classificação, mas conserva o nome de todos os arquivos.
    grupos = [('VÍDEOS SUSPEITOS', suspeitos),
              ('VÍDEOS NÃO SUSPEITOS', nao_suspeitos),
              ('VÍDEOS SEM CLASSIFICAÇÃO', restantes)]
    for nome_grupo, itens in grupos:
        if not itens:
            continue
        historia.append(Paragraph(f'{nome_grupo} ({len(itens)})', subtitulo))
        linhas = [[Paragraph(c, cabecalho) for c in ('Arquivo', 'Etapas executadas', 'Score', 'Tempo')]]
        for item in itens:
            caminho_relativo = str(item.get('caminho_relativo') or item.get('video') or 'Sem nome').replace('\\\\', '/')
            caminho_relativo = caminho_relativo.replace('\\', '/')
            caminho = Path(caminho_relativo)
            pasta = str(caminho.parent).replace('\\', '/') if str(caminho.parent) != '.' else ''
            arquivo_html = (f'<b>{escape(caminho.name)}</b>' +
                            (f'<br/><font color="#687986">{escape(pasta)}</font>' if pasta else ''))
            etapas = []
            if modo != 'AUDIO' and item.get('classificacao_risco_visual') is not None:
                if item.get('marca_dagua_confirmada') is True:
                    etapas.append('OCR: marca confirmada')
                elif item.get('status_triagem_marca') == 'DESATIVADA':
                    etapas.append('OCR: desativado')
                elif item.get('status_triagem_marca') is not None:
                    etapas.append('OCR: verificado')
                if item.get('score_spsl') is not None:
                    etapas.append('SPSL: parada' if item.get('spsl_interrompeu_effort') else 'SPSL: executado')
                else:
                    etapas.append('SPSL: não executado')
                if item.get('effort_executado') is True:
                    etapas.append('Effort: executado')
                elif item.get('effort_executado') is False:
                    etapas.append('Effort: dispensado')
            if modo != 'VISUAL' and item.get('status') in {'OK', 'OK_VISUAL_ONLY'}:
                etapas.append('DF-Arena-1B: executado' if item.get('score_audio') is not None
                              else 'DF-Arena-1B: não executado')
            if not etapas:
                etapas.append('Não concluído' if item.get('status') not in {'OK', 'OK_VISUAL_ONLY'}
                              else 'Verificar relatório individual')
            score = item.get(campo_score)
            try:
                score_texto = f'{float(score):.4f}'.replace('.', ',')
            except (TypeError, ValueError):
                score_texto = 'N/D'
            linhas.append([Paragraph(arquivo_html, celula),
                           Paragraph(escape(' | '.join(etapas)), celula),
                           texto_celula(score_texto),
                           texto_celula(segundos(item.get('tempo_total_video_segundos')))])
        tabela_itens = LongTable(linhas,
                                colWidths=[largura * 0.40, largura * 0.39,
                                           largura * 0.10, largura * 0.11], repeatRows=1)
        tabela_itens.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), azul)] + estilo_tabela))
        historia.append(tabela_itens)
    historia.append(Spacer(1, 10))
    historia.append(Paragraph('Para investigar um vídeo em detalhes, consulte os arquivos CSV e JSON '
                              'gerados na mesma pasta deste PDF. O relatório resume as etapas '
                              'executadas sem substituir as evidências individuais.', pequeno))
    doc.build(historia, onFirstPage=desenhar_pagina, onLaterPages=desenhar_pagina)
    return caminho_pdf


# Processa os vídeos do conjunto e guarda o progresso para retomada.
def run_local_dataset(continuar=False):
    gerenciador = GenericDatasetProcessor(str(INPUT_DIR))
    videos = gerenciador.find_videos()
    if not videos:
        print('\nNenhum vídeo encontrado em input/.')
        return
    nome_dataset = INPUT_DIR.resolve().name
    dataset_output = Path('output') / 'datasets' / nome_dataset
    if ANALYSIS_MODE == 'MULTIMODAL':
        dataset_output = dataset_output / ('com_rapidocr' if WATERMARK_SCREENING_ENABLED else 'sem_rapidocr')
    else:
        dataset_output = dataset_output / ('modo_' + slug_modo_analise(ANALYSIS_MODE))
    dataset_output.mkdir(parents=True, exist_ok=True)
    estado_path = dataset_output / 'estado_execucao.json'
    configuracao = configuracao_execucao()
    if continuar:
        if not estado_path.is_file():
            print('\nNão há execução compatível salva para retomar nesta modalidade e configuração.')
            return
        with estado_path.open(encoding='utf-8') as arquivo:
            estado = json.load(arquivo)
        if estado.get('configuracao') != configuracao:
            print('\nA configuração atual difere da execução interrompida. Ajuste o modo ou os parâmetros antes de retomar.')
            return
        registros_dataset = list(estado.get('registros_dataset') or [])
        resultados_pipeline = list(estado.get('resultados_pipeline') or [])
        assinaturas = dict(estado.get('assinaturas') or {})
        duracao_anterior = float(estado.get('duracao_acumulada_segundos') or 0)
        caminhos_atuais = {gerenciador.get_relative_path(v).as_posix() for v in videos}
        ausentes = [r.get('caminho_relativo') for r in registros_dataset if Path(r.get('caminho_relativo', '')).as_posix() not in caminhos_atuais]
        if ausentes:
            print('\nHá vídeos da execução anterior ausentes de input/. Restaure os arquivos antes de continuar.')
            return
        print(f'\nRetomada: {len(registros_dataset)} registro(s) recuperado(s).')
        gerenciador.save_results(registros_dataset, dataset_output)
        save_dataset_performance_csv(resultados_pipeline, dataset_output)
    else:
        if estado_path.exists() or (dataset_output / 'resultados_dataset.csv').exists():
            print('\nEsta pasta já contém resultados. Use [4] Continuar execução ou arquive a pasta de saída antes de iniciar um novo teste.')
            return
        registros_dataset, resultados_pipeline, assinaturas = ([], [], {})
        duracao_anterior = 0.0
        estado = {'configuracao': configuracao}
    confirmar_sem_audio = create_dataset_audio_confirmation() if ANALYSIS_MODE == 'MULTIMODAL' else None
    print(f'\nModalidade: {nome_modo_analise(ANALYSIS_MODE)}')
    print(f"RapidOCR: {('ATIVADO' if WATERMARK_SCREENING_ENABLED else 'DESATIVADO')}" if modo_inclui_visual() else 'Modalidade sem OCR')
    print(f'Vídeos identificados: {len(videos)} | Saída: {dataset_output}')
    cache_modelos = {}
    inicio_dataset = time.perf_counter()
    retomados = 0
    alterados = 0
    for indice, video_path in enumerate(videos, start=1):
        relative_path = gerenciador.get_relative_path(video_path)
        chave = relative_path.as_posix()
        atual = assinatura_video(video_path)
        anterior = next(
            (r for r in registros_dataset if Path(r.get('caminho_relativo', '')).as_posix() == chave),
            None
        )
        if anterior and anterior.get('status') in {'OK', 'OK_VISUAL_ONLY'} and (assinaturas.get(chave) == atual):
            retomados += 1
            print(f'[JÁ CONCLUÍDO {indice}/{len(videos)}] {chave}')
            continue
        if anterior:
            registros_dataset = [r for r in registros_dataset if Path(r.get('caminho_relativo', '')).as_posix() != chave]
            resultados_pipeline = [r for r in resultados_pipeline if str(r.get('video', '')) != str(video_path)]
            alterados += 1
        print('\n' + '#' * 70)
        print(f'VÍDEO [{indice}/{len(videos)}]: {video_path}')
        print('#' * 70)
        video_output = dataset_output / 'videos' / relative_path.with_suffix('')
        inicio_tentativa = time.perf_counter()
        try:
            resultado = run_pipeline(
                video_path,
                output_dir=video_output,
                audio_confirmation=confirmar_sem_audio,
                detector_cache=cache_modelos,
                analysis_mode=ANALYSIS_MODE
            )
            show_result(resultado)
            registro = registro_resultado(
                indice,
                video_path,
                relative_path,
                video_output,
                resultado,
                gerenciador
            )
        except Exception as erro:
            print(f'\nERRO: {type(erro).__name__}: {erro}')
            resultado = {
                'video': str(video_path),
                'status': 'ERRO',
                'erro': f'{type(erro).__name__}: {erro}',
                'performance': {
                    'application_wall_time_seconds': time.perf_counter() - inicio_tentativa,
                    'resource_configuration': RESOURCE_MANAGER.as_dict()
                }
            }
            registro = {
                'indice': indice,
                'video': video_path.name,
                'caminho_relativo': chave,
                'caminho_completo': str(video_path.resolve()),
                'classe_real': gerenciador.get_label(relative_path),
                'status': f'ERRO: {type(erro).__name__}: {erro}',
                'diretorio_saida': str(video_output)
            }
        registros_dataset.append(registro)
        resultados_pipeline.append(resultado_tempos_compacto(resultado))
        assinaturas[chave] = atual
        registros_dataset.sort(key=lambda item: int(item.get('indice', 0)))
        indice_por_video = {str(p): i for i, p in enumerate(videos)}
        resultados_pipeline.sort(key=lambda item: indice_por_video.get(
            str(item.get('video')),
            len(videos)
        ))
        estado.update({
            'registros_dataset': registros_dataset,
            'resultados_pipeline': resultados_pipeline,
            'assinaturas': assinaturas,
            'duracao_acumulada_segundos': duracao_anterior + (time.perf_counter() - inicio_dataset),
            'concluido': False
        })
        salvar_estado_execucao(estado_path, estado)
        gerenciador.save_results(registros_dataset, dataset_output)
        save_dataset_performance_csv(resultados_pipeline, dataset_output)
    salvo = gerenciador.save_results(registros_dataset, dataset_output)
    performance_csv = save_dataset_performance_csv(resultados_pipeline, dataset_output)
    print(f'\nReaproveitados sem reprocessar: {retomados} | Tentativas refeitas: {alterados}')
    fim_dataset = time.perf_counter()
    tempo_total_dataset = duracao_anterior + (fim_dataset - inicio_dataset)
    tempos_individuais = []
    for resultado in resultados_pipeline:
        performance = resultado.get('performance') or {}
        tempo_video = performance.get('application_wall_time_seconds')
        if tempo_video is not None:
            try:
                tempos_individuais.append(float(tempo_video))
            except (TypeError, ValueError):
                pass
    soma_tempos_videos = sum(tempos_individuais)
    quantidade_tempos_validos = len(tempos_individuais)
    tempo_medio_video = soma_tempos_videos / quantidade_tempos_validos if quantidade_tempos_validos > 0 else 0.0
    tempo_minimo_video = min(tempos_individuais) if tempos_individuais else 0.0
    tempo_maximo_video = max(tempos_individuais) if tempos_individuais else 0.0
    overhead_dataset = max(0.0, tempo_total_dataset - soma_tempos_videos)
    resumo = salvo['summary']
    resumo['modo_analise'] = ANALYSIS_MODE
    resumo['modo_analise_nome'] = nome_modo_analise(ANALYSIS_MODE)
    resumo['modo_desempenho'] = RESOURCE_MANAGER.mode
    resumo['rapidocr_ativado'] = WATERMARK_SCREENING_ENABLED if modo_inclui_visual() else None
    resumo['videos_reaproveitados_na_retomada'] = retomados
    resumo['limiar_binario_fusao_multimodal'] = (FUSION_SUSPICIOUS_THRESHOLD + FUSION_NOT_SUSPICIOUS_THRESHOLD) / 2.0
    resumo['tempo_total_dataset_segundos'] = float(tempo_total_dataset)
    resumo['tempo_total_dataset_formatado'] = formatar_duracao(tempo_total_dataset)
    resumo['soma_tempos_videos_segundos'] = float(soma_tempos_videos)
    resumo['soma_tempos_videos_formatada'] = formatar_duracao(soma_tempos_videos)
    resumo['overhead_dataset_segundos'] = float(overhead_dataset)
    resumo['overhead_dataset_formatado'] = formatar_duracao(overhead_dataset)
    resumo['videos_com_tempo_registrado'] = int(quantidade_tempos_validos)
    resumo['tempo_medio_video_segundos'] = float(tempo_medio_video)
    resumo['tempo_medio_video_formatado'] = formatar_duracao(tempo_medio_video)
    resumo['tempo_minimo_video_segundos'] = float(tempo_minimo_video)
    resumo['tempo_minimo_video_formatado'] = formatar_duracao(tempo_minimo_video)
    resumo['tempo_maximo_video_segundos'] = float(tempo_maximo_video)
    resumo['tempo_maximo_video_formatado'] = formatar_duracao(tempo_maximo_video)
    with open(salvo['summary_path'], 'w', encoding='utf-8') as arquivo:
        json.dump(resumo, arquivo, indent=4, ensure_ascii=False)
    estado.update({
        'registros_dataset': registros_dataset,
        'resultados_pipeline': resultados_pipeline,
        'assinaturas': assinaturas,
        'duracao_acumulada_segundos': tempo_total_dataset,
        'concluido': all((r.get('status') in {'OK', 'OK_VISUAL_ONLY'} for r in registros_dataset)) and len(registros_dataset) == len(videos)
    })
    salvar_estado_execucao(estado_path, estado)
    print('\n' + '=' * 78)
    print('PROCESSAMENTO DA PASTA FINALIZADO')
    print('=' * 78)
    print(f"Total: {resumo['total_videos']}")
    print(f"Processados: {resumo['processados_com_sucesso']}")
    print(f"Ignorados: {resumo.get('ignorados', 0)}")
    print(f"Erros: {resumo['erros']}")
    print(f'Modo de análise: {nome_modo_analise(ANALYSIS_MODE)}')
    print('\nRESULTADO MULTIMODAL FINAL')
    print('-' * 78)
    print(f"  SUSPEITO: {resumo.get('multimodais_suspeitos', 0)}")
    print(f"  NAO_SUSPEITO: {resumo.get('multimodais_nao_suspeitos', 0)}")
    print('\nMODALIDADE VISUAL')
    print('-' * 78)
    print(f"  SUSPEITO: {resumo.get('visuais_suspeitos', 0)}")
    print(f"  NAO_SUSPEITO: {resumo.get('visuais_nao_suspeitos', 0)}")
    metricas_visuais = resumo.get('metricas_visuais_rotuladas', {})
    if metricas_visuais.get('videos_avaliados', 0):
        balanced_accuracy = metricas_visuais.get('balanced_accuracy')
        print(f"  Avaliados com rótulo: {metricas_visuais['videos_avaliados']}")
        print(f"  Acurácia visual: {metricas_visuais['acuracia']:.1%}")
        print(f"  Sensibilidade fake: {metricas_visuais['sensibilidade_fake']:.1%}" if metricas_visuais.get('sensibilidade_fake') is not None else '  Sensibilidade fake: N/A')
        print(f"  Especificidade real: {metricas_visuais['especificidade_real']:.1%}" if metricas_visuais.get('especificidade_real') is not None else '  Especificidade real: N/A')
        print(f'  Balanced accuracy: {balanced_accuracy:.1%}' if balanced_accuracy is not None else '  Balanced accuracy: N/A (precisa de FAKE e REAL)')
        f1_suspeito = metricas_visuais.get('f1_suspeito')
        roc_auc_visual = metricas_visuais.get('roc_auc_visual')
        print(f'  F1 suspeito: {f1_suspeito:.1%}' if f1_suspeito is not None else '  F1 suspeito: N/A')
        print(f'  ROC-AUC visual: {roc_auc_visual:.1%}' if roc_auc_visual is not None else '  ROC-AUC visual: N/A (precisa de FAKE e REAL)')
    print('\nMODALIDADE DE ÁUDIO')
    print('-' * 78)
    print(f"  SUSPEITO: {resumo.get('audios_suspeitos', 0)}")
    print(f"  NAO_SUSPEITO: {resumo.get('audios_nao_suspeitos', 0)}")
    for titulo, campo in (
        ('SONORA', 'metricas_audio_rotuladas'),
        ('FUSÃO MULTIMODAL', 'metricas_multimodais_rotuladas')
    ):
        metricas = resumo.get(campo) or {}
        if metricas.get('videos_avaliados'):
            print(f"\nMÉTRICAS {titulo}: {metricas['videos_avaliados']} vídeos rotulados")
            print(f"  Acurácia: {metricas['acuracia']:.1%}")
            if metricas.get('balanced_accuracy') is not None:
                print(f"  Balanced accuracy: {metricas['balanced_accuracy']:.1%}")
            if metricas.get('f1_suspeito') is not None:
                print(f"  F1 suspeito: {metricas['f1_suspeito']:.1%}")
    print('\nTEMPO DE EXECUÇÃO DO DATASET')
    print('-' * 78)
    print(f'  Modo de desempenho: {RESOURCE_MANAGER.mode}')
    print(f'  Tempo total real do dataset: {formatar_duracao(tempo_total_dataset)} ({tempo_total_dataset:.3f} s)')
    print(f'  Soma dos tempos dos vídeos: {formatar_duracao(soma_tempos_videos)} ({soma_tempos_videos:.3f} s)')
    print(f'  Overhead entre vídeos/salvamentos: {formatar_duracao(overhead_dataset)} ({overhead_dataset:.3f} s)')
    print(f'  Vídeos com tempo registrado: {quantidade_tempos_validos}')
    print(f'  Tempo médio por vídeo: {formatar_duracao(tempo_medio_video)} ({tempo_medio_video:.3f} s)')
    print(f'  Vídeo mais rápido: {formatar_duracao(tempo_minimo_video)} ({tempo_minimo_video:.3f} s)')
    print(f'  Vídeo mais lento: {formatar_duracao(tempo_maximo_video)} ({tempo_maximo_video:.3f} s)')
    print('\nARQUIVOS GERADOS')
    print('-' * 78)
    print(f"Resultados: {salvo['csv']}")
    print(f"Resumo: {salvo['summary_path']}")
    print(f'Tempos: {performance_csv}')
    try:
        pdf_gerado = gerar_relatorio_pdf(registros_dataset, resumo, dataset_output)
        print(f'Relatório PDF: {pdf_gerado}')
    except Exception as erro_pdf:
        print(f'Relatório PDF não gerado: {type(erro_pdf).__name__}: {erro_pdf}')
        print('Os resultados CSV e JSON foram preservados. Para habilitar o PDF: pip install reportlab')


def show_analysis_mode_menu():
    print('\n' + '=' * 70)
    print('SELECIONE A MODALIDADE DE ANÁLISE')
    print('=' * 70)
    print('[1] VISUAL — OCR, SPSL e Effort')
    print('[2] SONORA — DF-Arena-1B')
    print('[3] FUSÃO MULTIMODAL — visual + sonora')
    print('[0] Sair')
    return input('\nEscolha: ').strip()


def configure_analysis_mode(permitir_sair=True):
    global ANALYSIS_MODE
    mapping = {'1': 'VISUAL', '2': 'AUDIO', '3': 'MULTIMODAL'}
    while True:
        opcao = show_analysis_mode_menu()
        if opcao == '0':
            if permitir_sair:
                return False
            print('Selecione uma modalidade para continuar.')
            continue
        if opcao not in mapping:
            print('Opção inválida.')
            continue
        ANALYSIS_MODE = mapping[opcao]
        configurar_rapidocr()
        print(f'\nModalidade selecionada: {nome_modo_analise(ANALYSIS_MODE)}')
        return True


def aplicar_modo_desempenho(modo):
    global RESOURCE_MANAGER
    global PERFORMANCE_MODE
    novo = ResourceManager(modo)
    novo.apply_environment_limits()
    novo.apply_runtime_limits()
    RESOURCE_MANAGER = novo
    PERFORMANCE_MODE = novo.mode
    print('\n' + '=' * 70)
    print('CONFIGURAÇÃO DE DESEMPENHO ATUALIZADA')
    print('=' * 70)
    RESOURCE_MANAGER.print_summary()
    print('A nova configuração será usada na próxima análise. Não é necessário reiniciar o programa.')


def show_performance_menu():
    print(f'\nCONFIGURAÇÃO DE DESEMPENHO — atual: {RESOURCE_MANAGER.mode}')
    print('[1] SEQUENCIAL — um ramo pesado por vez')
    print('[2] ECONOMICO — maior reserva de recursos para o sistema')
    print('[3] BALANCEADO — paralelismo controlado [PADRÃO]')
    print('[4] DESEMPENHO — maior uso de CPU')
    print('[0] Voltar')
    return input('\nEscolha: ').strip()


def configure_performance_mode():
    mapping = {'1': 'SEQUENCIAL', '2': 'ECONOMICO', '3': 'BALANCEADO', '4': 'DESEMPENHO'}
    while True:
        opcao = show_performance_menu()
        if opcao == '0':
            return
        modo = mapping.get(opcao)
        if modo is None:
            print('\nOpção inválida.')
            continue
        aplicar_modo_desempenho(modo)
        return


def listar_entrada():
    pasta = INPUT_DIR
    pasta.mkdir(parents=True, exist_ok=True)
    extensoes = GenericDatasetProcessor.VIDEO_EXTENSIONS
    return sorted(
        (c for c in pasta.rglob('*') if c.is_file() and c.suffix.lower() in extensoes),
        key=lambda c: str(c).casefold()
    )


def run_input(continuar=False):
    videos = listar_entrada()
    if not videos:
        print('\nNenhum vídeo encontrado em input/ ou nas suas subpastas.')
        return
    print(f'\nForam encontrados {len(videos)} vídeo(s) em input/:')
    for video in videos[:10]:
        print(f'  - {video.relative_to(INPUT_DIR)}')
    if len(videos) > 10:
        print(f'  ... e mais {len(videos) - 10} arquivo(s).')
    resposta = input('\nPressione ENTER para ' + ('continuar' if continuar else 'iniciar') + ' as verificações (ou digite 0 para cancelar): ').strip()
    if resposta:
        print('\nExecução cancelada. Nenhum vídeo foi processado.')
        return
    run_local_dataset(continuar=continuar)


def show_menu():
    videos = listar_entrada()
    print('\n' + '=' * 70)
    print('PIPELINE MULTIMODAL DE DETECÇÃO DE DEEPFAKES')
    print('=' * 70)
    print(f'Entrada: {INPUT_DIR}/ (identificação automática)')
    print(f'Vídeos encontrados em input/: {len(videos)}')
    if len(videos) == 1:
        print(f'Arquivo: {videos[0].relative_to(INPUT_DIR)}')
    elif len(videos) > 1:
        print(f'Primeiro arquivo: {videos[0].relative_to(INPUT_DIR)}')
        print(f'Último arquivo: {videos[-1].relative_to(INPUT_DIR)}')
    print(f'Modalidade atual: {nome_modo_analise(ANALYSIS_MODE)}')
    print(f'Modo de desempenho: {RESOURCE_MANAGER.mode}')
    print(f'\n[1] Analisar {len(videos)} vídeo(s) de input/')
    print('[2] Configuração de desempenho')
    print('[3] Alterar modalidade de análise')
    print('[4] Continuar execução interrompida')
    print('[0] Sair')
    return input('\nEscolha: ').strip()


# O menu apresenta as opções de análise e de retomada.
def main():
    os.chdir(Path(__file__).resolve().parent)
    print(f'\nModo de desempenho padrão: {DEFAULT_PERFORMANCE_MODE}')
    if not configure_analysis_mode(permitir_sair=True):
        print('\nPipeline encerrada.')
        return
    while True:
        opcao = show_menu()
        try:
            if opcao == '1':
                run_input()
            elif opcao == '2':
                configure_performance_mode()
            elif opcao == '3':
                configure_analysis_mode(permitir_sair=False)
            elif opcao == '4':
                run_input(continuar=True)
            elif opcao == '0':
                print('\nPipeline encerrada.')
                break
            else:
                print('\nOpção inválida.')
        except Exception as erro:
            print(f'\nERRO: {type(erro).__name__}: {erro}')


if __name__ == '__main__':
    main()

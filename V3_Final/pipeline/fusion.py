# Combina os indicadores das modalidades visual e sonora.
import json
from pathlib import Path
from datetime import datetime, timezone
from statistics import median


# Reúne os resultados visual e sonoro sem mudar os detectores.
class MultimodalFusionAnalyzer:

    def __init__(
        self,
        peso_visual=0.50,
        peso_audio=0.50,
        limite_suspeito=0.67,
        limite_nao_suspeito=0.47,
        max_frames_suspeitos=5,
        max_timestamps_suspeitos=12
    ):
        self.peso_visual = float(peso_visual)
        self.peso_audio = float(peso_audio)
        self.limite_suspeito = float(limite_suspeito)
        self.limite_nao_suspeito = float(limite_nao_suspeito)
        self.max_frames_suspeitos = int(max_frames_suspeitos)
        self.max_timestamps_suspeitos = int(max_timestamps_suspeitos)
        self.minimo_modalidades_decisao = 1
        if self.peso_visual <= 0 or self.peso_audio <= 0:
            raise ValueError('Os pesos da fusão multimodal devem ser positivos.')
        if not 0 <= self.limite_nao_suspeito < self.limite_suspeito <= 1:
            raise ValueError('Limiares de fusão inválidos.')

    @staticmethod
    def _score(valor):
        if valor is None:
            return None
        try:
            valor = float(valor)
        except (TypeError, ValueError):
            return None
        if 0 <= valor <= 1:
            return valor
        return None

    @staticmethod
    def _clamp(valor):
        return max(0.0, min(1.0, float(valor)))

    @staticmethod
    def _tempo(segundos):
        if segundos is None:
            return 'N/A'
        try:
            segundos = float(segundos)
        except (TypeError, ValueError):
            return 'N/A'
        minuto = int(segundos // 60)
        resto = segundos - minuto * 60
        return f'{minuto:02d}:{resto:05.2f}'

    @staticmethod
    def _media(valores):
        valores = [float(valor) for valor in valores if valor is not None]
        if not valores:
            return None
        return sum(valores) / len(valores)

    @staticmethod
    def _formatar_percentual_opcional(valor):
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            return 'N/A'
        if not 0.0 <= numero <= 1.0:
            return 'N/A'
        return f'{numero:.2%}'

    @staticmethod
    def _upper(valor):
        if valor is None:
            return None
        return str(valor).strip().upper()

    def _estatisticas_visuais(self, visual_data):
        visual_data = visual_data or {}
        threshold = self._score(visual_data.get('threshold')) or 0.5
        itens = []
        for item in visual_data.get('frame_scores', []):
            score = self._score(item.get('visual_score'))
            if score is None:
                continue
            try:
                timestamp = float(item.get('timestamp'))
            except (TypeError, ValueError):
                timestamp = None
            itens.append({
                'score': score,
                'timestamp': timestamp,
                'frame_index': item.get('frame_index')
            })
        if not itens:
            return {
                'threshold': threshold,
                'quantidade_frames_com_score': 0,
                'media_frames': None,
                'mediana_frames': None,
                'fracao_frames_suspeitos': None,
                'fracao_frames_muito_suspeitos': None,
                'maior_fragmento_suspeito_segundos': 0.0,
                'fragmentos_suspeitos': []
            }
        scores = [item['score'] for item in itens]
        quantidade_suspeitos = sum((item['score'] >= threshold for item in itens))
        quantidade_muito_suspeitos = sum((item['score'] >= 0.8 for item in itens))
        fracao_suspeita = quantidade_suspeitos / len(itens)
        fracao_muito_suspeita = quantidade_muito_suspeitos / len(itens)
        fragmentos = self._fragmentos_visuais(itens, threshold)
        maior_fragmento = max((item['duracao'] for item in fragmentos), default=0.0)
        return {
            'threshold': threshold,
            'quantidade_frames_com_score': len(itens),
            'media_frames': sum(scores) / len(scores),
            'mediana_frames': float(median(scores)),
            'fracao_frames_suspeitos': fracao_suspeita,
            'fracao_frames_muito_suspeitos': fracao_muito_suspeita,
            'maior_fragmento_suspeito_segundos': maior_fragmento,
            'fragmentos_suspeitos': fragmentos
        }

    def _fragmentos_visuais(self, itens, threshold):
        candidatos = sorted(
            [item for item in itens if item['timestamp'] is not None and item['score'] >= threshold],
            key=lambda item: item['timestamp']
        )
        if not candidatos:
            return []
        grupos = [[candidatos[0]]]
        for item in candidatos[1:]:
            anterior = grupos[-1][-1]
            if item['timestamp'] - anterior['timestamp'] <= 1.0:
                grupos[-1].append(item)
            else:
                grupos.append([item])
        saida = []
        for grupo in grupos:
            inicio = grupo[0]['timestamp']
            fim = grupo[-1]['timestamp']
            scores = [item['score'] for item in grupo]
            duracao = max(0.0, fim - inicio)
            saida.append({
                'inicio': inicio,
                'fim': fim,
                'inicio_formatado': self._tempo(inicio),
                'fim_formatado': self._tempo(fim),
                'duracao': duracao,
                'quantidade_frames': len(grupo),
                'score_medio': sum(scores) / len(scores),
                'score_maximo': max(scores),
                'fragmento_relevante': duracao >= 1.5 or len(grupo) >= 4
            })
        return sorted(saida, key=lambda item: (item['duracao'], item['score_medio']), reverse=True)

    def _estatisticas_audio(self, audio_data):
        audio_data = audio_data or {}
        threshold = self._score(audio_data.get('threshold')) or 0.5
        itens = []
        for item in audio_data.get('segment_scores', []):
            score = self._score(item.get('spoof_score'))
            if score is None:
                continue
            inicio = item.get('start_time', item.get('start'))
            fim = item.get('end_time', item.get('end'))
            try:
                inicio = float(inicio) if inicio is not None else None
            except (TypeError, ValueError):
                inicio = None
            try:
                fim = float(fim) if fim is not None else None
            except (TypeError, ValueError):
                fim = None
            itens.append({'score': score, 'inicio': inicio, 'fim': fim})
        if not itens:
            return {
                'threshold': threshold,
                'quantidade_janelas_com_score': 0,
                'media_janelas': None,
                'mediana_janelas': None,
                'fracao_janelas_suspeitas': None,
                'maior_fragmento_suspeito_segundos': 0.0,
                'fragmentos_suspeitos': []
            }
        scores = [item['score'] for item in itens]
        quantidade_suspeitas = sum((item['score'] >= threshold for item in itens))
        fracao_suspeita = quantidade_suspeitas / len(itens)
        fragmentos = self._fragmentos_audio(itens, threshold)
        maior_fragmento = max((item['duracao'] for item in fragmentos), default=0.0)
        return {
            'threshold': threshold,
            'quantidade_janelas_com_score': len(itens),
            'media_janelas': sum(scores) / len(scores),
            'mediana_janelas': float(median(scores)),
            'fracao_janelas_suspeitas': fracao_suspeita,
            'maior_fragmento_suspeito_segundos': maior_fragmento,
            'fragmentos_suspeitos': fragmentos
        }

    def _fragmentos_audio(self, itens, threshold):
        candidatos = sorted(
            [item for item in itens if item['inicio'] is not None and item['fim'] is not None and (item['score'] >= threshold)],
            key=lambda item: item['inicio']
        )
        if not candidatos:
            return []
        grupos = [[candidatos[0]]]
        for item in candidatos[1:]:
            anterior = grupos[-1][-1]
            if item['inicio'] - anterior['fim'] <= 0.25:
                grupos[-1].append(item)
            else:
                grupos.append([item])
        saida = []
        for grupo in grupos:
            inicio = grupo[0]['inicio']
            fim = grupo[-1]['fim']
            scores = [item['score'] for item in grupo]
            saida.append({
                'inicio': inicio,
                'fim': fim,
                'inicio_formatado': self._tempo(inicio),
                'fim_formatado': self._tempo(fim),
                'duracao': max(0.0, fim - inicio),
                'quantidade_janelas': len(grupo),
                'score_medio': sum(scores) / len(scores),
                'score_maximo': max(scores)
            })
        return sorted(saida, key=lambda item: (item['duracao'], item['score_medio']), reverse=True)

    # Organiza os indicadores recebidos do ramo visual.
    def _visual(self, visual_data):
        visual_data = visual_data or {}
        original = self._score(visual_data.get('visual_score'))
        estatisticas = self._estatisticas_visuais(visual_data)
        estatisticas_detector = visual_data.get('visual_statistics', {}) or {}
        metodo_score = str(visual_data.get('visual_score_method', '') or '').strip().upper()
        metodo_agregacao = str(visual_data.get('aggregation_method', '') or '').strip().lower()
        usa_score_visual_temporal = bool(metodo_score == 'VISUAL_TEMPORAL' or metodo_agregacao == 'temporal_multisignal')
        usa_score_visual_ensemble_sbi_npr = bool(metodo_score == 'VISUAL_ENSEMBLE_SBI_NPR' or metodo_agregacao == 'sbi_npr_agreement')
        usa_score_visual_ensemble_effort_commfor = bool(metodo_score == 'VISUAL_ENSEMBLE_EFFORT_COMMFOR_SCREENING' or metodo_agregacao == 'effort_commfor_uniform_mean_consensus')
        usa_score_visual_cauda_superior = bool(metodo_score == 'VISUAL_EFFORT_UPPER_TAIL' or metodo_agregacao == 'effort_p90_top25_equal')
        usa_score_visual_media = bool(metodo_score == 'VISUAL_EFFORT_MEAN' or metodo_agregacao == 'effort_uniform_mean')
        usa_score_visual_ensemble = bool(usa_score_visual_ensemble_sbi_npr or usa_score_visual_ensemble_effort_commfor)
        usa_score_visual_direto = bool(usa_score_visual_temporal or usa_score_visual_ensemble or usa_score_visual_cauda_superior or usa_score_visual_media)
        if original is None:
            return {
                'disponivel': False,
                'usada_na_fusao': False,
                'score_original': None,
                'score_suspeita': None,
                'peso_base': self.peso_visual,
                'peso_normalizado': 0.0,
                'nivel_alerta': 'INDETERMINADO',
                'alerta_critico': False,
                'confianca_modal': 0.0,
                'estatisticas_temporais': estatisticas,
                'estatisticas_detector': estatisticas_detector,
                'visual_score_method': metodo_score or None
            }
        if usa_score_visual_direto:
            ajustado = original
        else:
            media = estatisticas.get('media_frames')
            persistencia_legada = estatisticas.get('fracao_frames_suspeitos')
            if media is None or persistencia_legada is None:
                ajustado = original
            else:
                ajustado = self._clamp(0.45 * original + 0.35 * media + 0.2 * persistencia_legada)
        persistencia_40 = self._score(estatisticas_detector.get('persistencia_40'))
        if persistencia_40 is None:
            persistencia_40 = estatisticas.get('fracao_frames_suspeitos') or 0.0
        fracao_40 = self._score(estatisticas_detector.get('fracao_acima_40'))
        if fracao_40 is None:
            fracao_40 = 0.0
        fracao_50 = self._score(estatisticas_detector.get('fracao_acima_50'))
        if fracao_50 is None:
            fracao_50 = estatisticas.get('fracao_frames_suspeitos') or 0.0
        fracao_muito = self._score(estatisticas_detector.get('fracao_acima_80'))
        if fracao_muito is None:
            fracao_muito = estatisticas.get('fracao_frames_muito_suspeitos') or 0.0
        maior_fragmento = (estatisticas_detector.get('maior_fragmento_40') or {}).get('duracao_efetiva')
        if maior_fragmento is None:
            maior_fragmento = estatisticas.get('maior_fragmento_suspeito_segundos') or 0.0
        classificacao_original = self._upper(visual_data.get('classification'))
        if usa_score_visual_ensemble or usa_score_visual_cauda_superior or usa_score_visual_media:
            motivo_fusao_visual = self._upper(visual_data.get('fusion_reason'))
            motivos_concordancia_forte = {
                'CONCORDANCIA_ENTRE_SBI_E_NPR',
                'CONCORDANCIA_EFFORT_COMMFOR'
            }
            critico = bool(classificacao_original == 'SUSPEITO' and ajustado >= 0.8 and (usa_score_visual_cauda_superior or usa_score_visual_media or motivo_fusao_visual in motivos_concordancia_forte))
            if critico:
                nivel = 'CRITICO'
            elif classificacao_original == 'SUSPEITO' or ajustado >= 0.6:
                nivel = 'ALTO'
            elif ajustado > 0.4 and ajustado < 0.6:
                nivel = 'MODERADO'
            else:
                nivel = 'BAIXO'
        else:
            critico = bool(ajustado >= 0.8 and persistencia_40 >= 0.7 and (fracao_50 >= 0.35))
            if critico:
                nivel = 'CRITICO'
            elif classificacao_original == 'SUSPEITO' or ajustado >= 0.55 or fracao_muito >= 0.2:
                nivel = 'ALTO'
            elif ajustado > 0.35 or persistencia_40 >= 0.35 or maior_fragmento >= 1.5:
                nivel = 'MODERADO'
            else:
                nivel = 'BAIXO'
        confianca_detector = self._score(visual_data.get(
            'classification_confidence',
            visual_data.get('visual_confidence')
        ))
        if confianca_detector is not None:
            confianca = confianca_detector
        else:
            consistencia = self._clamp(1 - abs(original - ajustado))
            amostra = self._clamp((estatisticas.get('quantidade_frames_com_score') or 0) / 30)
            confianca = self._clamp(0.7 * consistencia + 0.3 * amostra)
        return {
            'disponivel': True,
            'usada_na_fusao': True,
            'score_original': original,
            'score_suspeita': ajustado,
            'score_suspeita_ajustado': ajustado,
            'score_visual_baseline_top25': self._score(visual_data.get('visual_score_baseline_top25')),
            'classificacao_original': visual_data.get('classification'),
            'visual_score_method': metodo_score or None,
            'peso_base': self.peso_visual,
            'nivel_alerta': nivel,
            'alerta_critico': critico,
            'confianca_modal': confianca,
            'consistencia_visual': self._score(visual_data.get('visual_consistency')),
            'frames_analisados': visual_data.get('frames_analyzed'),
            'rostos_analisados': visual_data.get('faces_analyzed'),
            'metodo_agregacao': visual_data.get('aggregation_method'),
            'estatisticas_temporais': estatisticas,
            'estatisticas_detector': estatisticas_detector,
            'score_sbi': self._score(visual_data.get('score_sbi')),
            'classificacao_sbi': visual_data.get('classification_sbi'),
            'score_npr': self._score(visual_data.get('score_npr')),
            'classificacao_npr': visual_data.get('classification_npr'),
            'score_effort': self._score(visual_data.get('score_effort')),
            'classificacao_effort': visual_data.get('classification_effort'),
            'score_effort_media': self._score(visual_data.get('score_effort_mean')),
            'score_community_forensics': self._score(visual_data.get('score_commfor')),
            'classificacao_community_forensics': visual_data.get('classification_commfor'),
            'sbi_influencia_decisao': visual_data.get('sbi_influencia_decisao'),
            'motivo_fusao_visual': visual_data.get('fusion_reason'),
            'observacao': 'Score visual integrado ao score sonoro com pesos normalizados.'
        }

    # Organiza os indicadores recebidos do ramo sonoro.
    def _audio(self, audio_data):
        audio_data = audio_data or {}
        original = self._score(audio_data.get('spoof_score'))
        estatisticas = self._estatisticas_audio(audio_data)
        if original is None:
            return {
                'disponivel': False,
                'usada_na_fusao': False,
                'score_original': None,
                'score_suspeita': None,
                'peso_base': self.peso_audio,
                'peso_normalizado': 0.0,
                'nivel_alerta': 'INDETERMINADO',
                'alerta_critico': False,
                'confianca_modal': 0.0,
                'estatisticas_temporais': estatisticas
            }
        media = estatisticas.get('media_janelas')
        persistencia = estatisticas.get('fracao_janelas_suspeitas')
        if media is None or persistencia is None:
            ajustado = original
        else:
            ajustado = self._clamp(0.45 * original + 0.35 * media + 0.2 * persistencia)
        persistencia = persistencia or 0.0
        critico = ajustado >= 0.75 and persistencia >= 0.4
        if critico:
            nivel = 'CRITICO'
        elif ajustado >= 0.6 or persistencia >= 0.25:
            nivel = 'ALTO'
        elif ajustado >= 0.4 or original >= 0.6:
            nivel = 'MODERADO'
        else:
            nivel = 'BAIXO'
        consistencia = self._clamp(1 - abs(original - ajustado))
        amostra = self._clamp((estatisticas.get('quantidade_janelas_com_score') or 0) / 10)
        confianca = self._clamp(0.7 * consistencia + 0.3 * amostra)
        return {
            'disponivel': True,
            'usada_na_fusao': True,
            'score_original': original,
            'score_suspeita': ajustado,
            'score_suspeita_ajustado': ajustado,
            'bonafide_score': self._score(audio_data.get('bonafide_score')),
            'classificacao_original': audio_data.get('classification'),
            'peso_base': self.peso_audio,
            'nivel_alerta': nivel,
            'alerta_critico': critico,
            'confianca_modal': confianca,
            'janelas_analisadas': audio_data.get('segments_analyzed'),
            'metodo_agregacao': audio_data.get('aggregation_method'),
            'estatisticas_temporais': estatisticas,
            'observacao': 'O score de spoof original é mantido, mas a fusão considera média e persistência das janelas para distinguir picos isolados de voz sintética recorrente.'
        }

    def _pesos(self, modalidades):
        usadas = [nome for nome, dados in modalidades.items() if dados.get('usada_na_fusao')]
        soma = sum((modalidades[nome].get('peso_base', 0.0) for nome in usadas))
        for nome, dados in modalidades.items():
            if nome in usadas and soma > 0:
                dados['peso_normalizado'] = dados.get('peso_base', 0.0) / soma
            else:
                dados['peso_normalizado'] = 0.0
        return usadas

    # Combina os scores válidos com os pesos normalizados.
    def _score_final(self, modalidades):
        parcelas = []
        for nome, dados in modalidades.items():
            if not dados.get('usada_na_fusao'):
                continue
            score = self._score(dados.get('score_suspeita'))
            peso = float(dados.get('peso_normalizado', 0.0))
            if score is None or peso <= 0:
                continue
            contribuicao = score * peso
            dados['contribuicao_score_final'] = contribuicao
            parcelas.append({
                'modalidade': nome,
                'score': score,
                'peso': peso,
                'contribuicao': contribuicao,
                'confianca_modal': dados.get('confianca_modal', 0.0)
            })
        if not parcelas:
            return (None, [])
        score_final = sum((item['contribuicao'] for item in parcelas))
        return (self._clamp(score_final), parcelas)

    def _concordancia(self, parcelas):
        if not parcelas:
            return (0.0, 1.0)
        if len(parcelas) == 1:
            return (1.0, 0.0)
        scores = [item['score'] for item in parcelas]
        conflito = self._clamp(max(scores) - min(scores))
        concordancia = self._clamp(1 - conflito)
        return (concordancia, conflito)

    def _confianca(self, score_final, classificacao, parcelas, concordancia, conflito, criticas):
        if score_final is None or not parcelas:
            return 0.0
        qualidade = self._media([self._clamp(item.get('confianca_modal', 0.0)) for item in parcelas]) or 0.0
        if classificacao == 'NAO_SUSPEITO':
            margem = self._clamp((self.limite_nao_suspeito - score_final) / max(self.limite_nao_suspeito, 1e-06))
            return self._clamp(0.45 * margem + 0.3 * concordancia + 0.25 * qualidade)
        if classificacao == 'SUSPEITO':
            if criticas:
                conf_criticas = [item.get('confianca_modal', 0.0) for item in parcelas if item['modalidade'] in criticas]
                base = max(conf_criticas, default=0.6)
                return self._clamp(0.55 * base + 0.25 * qualidade + 0.2 * (1 - conflito))
            margem = self._clamp((score_final - self.limite_suspeito) / max(1 - self.limite_suspeito, 1e-06))
            return self._clamp(0.45 * margem + 0.3 * concordancia + 0.25 * qualidade)
        largura = max(self.limite_suspeito - self.limite_nao_suspeito, 1e-06)
        ambiguidade = self._clamp(1 - abs(score_final - 0.5) / (largura / 2))
        return self._clamp(0.4 * ambiguidade + 0.35 * conflito + 0.25 * qualidade)

    def _mapa_frames(self, frame_data):
        mapa = {}
        for frame in (frame_data or {}).get('frames', []):
            indice = frame.get('frame_index')
            if indice is None:
                continue
            mapa[indice] = {
                'timestamp': frame.get('timestamp'),
                'frame_path': str(frame.get('path')) if frame.get('path') is not None else None
            }
        return mapa

    def _mapa_faces(self, face_data):
        mapa = {}
        for face in (face_data or {}).get('faces', []):
            indice = face.get('frame_index')
            if indice is None:
                continue
            conf = self._score(face.get('confidence')) or 0.0
            if indice not in mapa or conf > mapa[indice].get('confidence', 0.0):
                mapa[indice] = {
                    'face_path': str(face.get('face_path')) if face.get('face_path') is not None else None,
                    'frame_path': str(face.get('frame_path')) if face.get('frame_path') is not None else None,
                    'confidence': conf,
                    'bounding_box': face.get('bounding_box')
                }
        return mapa

    def _frames_suspeitos(self, visual_data, frame_data, face_data):
        visual_data = visual_data or {}
        threshold = self._score(visual_data.get('threshold')) or 0.5
        frames = self._mapa_frames(frame_data)
        faces = self._mapa_faces(face_data)
        face_sbi = {}
        for item in visual_data.get('face_scores', []):
            indice = item.get('frame_index')
            score = self._score(item.get('sbi_score'))
            if indice is None or score is None:
                continue
            if indice not in face_sbi or score > face_sbi[indice]['score']:
                face_sbi[indice] = {
                    'score': score,
                    'face_path': str(item.get('face_path')) if item.get('face_path') is not None else None
                }
        saida = []
        for item in visual_data.get('frame_scores', []):
            score = self._score(item.get('visual_score'))
            if score is None or score < threshold:
                continue
            indice = item.get('frame_index')
            base_frame = frames.get(indice, {})
            base_face = faces.get(indice, {})
            sbi = face_sbi.get(indice, {})
            timestamp = item.get('timestamp', base_frame.get('timestamp'))
            saida.append({
                'frame_index': indice,
                'timestamp': timestamp,
                'timestamp_formatado': self._tempo(timestamp),
                'visual_score': score,
                'frame_path': base_frame.get('frame_path') or base_face.get('frame_path'),
                'face_path': sbi.get('face_path') or base_face.get('face_path'),
                'face_sbi_score': sbi.get('score'),
                'bounding_box': base_face.get('bounding_box')
            })
        return sorted(
            saida,
            key=lambda item: item['visual_score'],
            reverse=True
        )[:self.max_frames_suspeitos]

    def _timestamps_audio(self, audio_data):
        audio_data = audio_data or {}
        threshold = self._score(audio_data.get('threshold')) or 0.5
        saida = []
        for item in audio_data.get('segment_scores', []):
            score = self._score(item.get('spoof_score'))
            if score is None or score < threshold:
                continue
            inicio = item.get('start_time', item.get('start'))
            fim = item.get('end_time', item.get('end'))
            saida.append({
                'modalidade': 'AUDIO',
                'inicio': inicio,
                'fim': fim,
                'inicio_formatado': self._tempo(inicio),
                'fim_formatado': self._tempo(fim),
                'score_suspeita': score,
                'descricao': 'Trecho com score elevado de voz sintética/spoof.'
            })
        return sorted(saida, key=lambda item: item['score_suspeita'], reverse=True)

    # Aplica o limiar definido para decidir a classificação.
    def _classificar(self, score_final):
        if score_final is None:
            return ('INCONCLUSIVO', 'SEM_EVIDENCIA_MULTIMODAL_VALIDADA', [])
        limiar = (self.limite_nao_suspeito + self.limite_suspeito) / 2.0
        if score_final >= limiar:
            return ('SUSPEITO', 'SCORE_ACIMA_DO_LIMIAR_BINARIO', [])
        return ('NAO_SUSPEITO', 'SCORE_ABAIXO_DO_LIMIAR_BINARIO', [])

    def _timestamps(self, frames, audio_data):
        eventos = []
        for frame in frames:
            eventos.append({
                'modalidade': 'VISUAL',
                'inicio': frame.get('timestamp'),
                'fim': frame.get('timestamp'),
                'inicio_formatado': frame.get('timestamp_formatado'),
                'fim_formatado': frame.get('timestamp_formatado'),
                'score_suspeita': frame.get('visual_score'),
                'frame_index': frame.get('frame_index'),
                'frame_path': frame.get('frame_path'),
                'face_path': frame.get('face_path'),
                'descricao': 'Frame com indicador visual elevado.'
            })
        eventos.extend(self._timestamps_audio(audio_data))
        return sorted(
            eventos,
            key=lambda e: e.get('score_suspeita') or 0.0,
            reverse=True
        )[:self.max_timestamps_suspeitos]

    def _alertas(self, modalidades):
        alertas = []
        for nome in ('visual', 'audio'):
            info = modalidades[nome]
            if info.get('disponivel'):
                alertas.append({
                    'modalidade': nome.upper(),
                    'nivel': info.get('nivel_alerta'),
                    'critico': bool(info.get('alerta_critico')),
                    'mensagem': f"Score bruto={info.get('score_original'):.4f}; score ajustado={info.get('score_suspeita'):.4f}"
                })
        ordem = {'CRITICO': 4, 'ALTO': 3, 'MODERADO': 2, 'BAIXO': 1, 'INDETERMINADO': 0}
        return sorted(alertas, key=lambda a: ordem.get(a.get('nivel'), 0), reverse=True)

    def _interpretacao(self, classe, motivo, modalidades, concordancia, conflito):
        usados = [x for x in ('visual', 'audio') if modalidades[x].get('usada_na_fusao')]
        if not usados:
            return 'Não há indicadores multimodais válidos para uma classificação.'
        descricoes = ', '.join((f"{nome}={modalidades[nome]['score_suspeita']:.4f}" for nome in usados))
        return f'Classificação multimodal: {classe}. Indicadores utilizados: {descricoes}. Concordância={concordancia:.4f}; conflito={conflito:.4f}. A análise multimodal não verifica a veracidade das falas.'

    @staticmethod
    def _relatorio_txt(r):
        lines = [
            'RELATÓRIO DE DETECÇÃO MULTIMODAL',
            '=' * 58,
            f"Classificação: {r['classification']}",
            f"Score: {r['suspicion_score']}",
            f"Confiança: {r['confidence']:.4f}",
            f"Modalidades: {', '.join(r['modalities_used']) or 'nenhuma'}",
            f"Interpretação: {r['interpretation']}",
            '',
            'INDICADORES'
        ]
        for nome, dados in r['modalities'].items():
            lines.append(f"{nome.upper()}: original={dados.get('score_original')} ajustado={dados.get('score_suspeita')} peso={dados.get('peso_normalizado')}")
        lines.append(f"Frames de interesse: {len(r['suspicious_frames'])}")
        lines.append(f"Timestamps de interesse: {len(r['suspicious_timestamps'])}")
        return '\n'.join(lines) + '\n'

    # Monta a saída da fusão e registra as evidências disponíveis.
    def process(
        self,
        visual_data,
        audio_data,
        frame_data=None,
        face_data=None,
        output_dir='output/fusion'
    ):
        pasta = Path(output_dir)
        pasta.mkdir(parents=True, exist_ok=True)
        modalidades = {'visual': self._visual(visual_data), 'audio': self._audio(audio_data)}
        usadas = self._pesos(modalidades)
        score, parcelas = self._score_final(modalidades)
        concordancia, conflito = self._concordancia(parcelas)
        classe, motivo, criticas = self._classificar(score)
        confianca = self._confianca(score, classe, parcelas, concordancia, conflito, criticas)
        frames = self._frames_suspeitos(visual_data, frame_data, face_data)
        timestamps = self._timestamps(frames, audio_data)
        resultado = {
            'status': 'OK',
            'method': 'FUSAO_MULTIMODAL_TARDIA',
            'classification': classe,
            'decision_reason': motivo,
            'critical_modalities': criticas,
            'suspicion_score': score,
            'suspicion_score_is_probability': False,
            'confidence': confianca,
            'confidence_is_probability': False,
            'agreement_score': concordancia,
            'conflict_score': conflito,
            'decision_thresholds': {
                'nao_suspeito_max': self.limite_nao_suspeito,
                'suspeito_min': self.limite_suspeito,
                'limiar_binario': (self.limite_suspeito + self.limite_nao_suspeito) / 2
            },
            'base_weights': {'visual': self.peso_visual, 'audio': self.peso_audio},
            'fusion_policy': {
                'strategy': 'FUSAO_TARDIA_MULTIMODAL',
                'normalizacao': 'Soma dos pesos das modalidades disponíveis',
                'limiar_binario': (self.limite_suspeito + self.limite_nao_suspeito) / 2
            },
            'modalities_used': usadas,
            'modalities_used_count': len(parcelas),
            'minimum_modalities_for_decision': self.minimo_modalidades_decisao,
            'modalities': modalidades,
            'individual_alerts': self._alertas(modalidades),
            'suspicious_timestamps': timestamps,
            'suspicious_frames': frames,
            'interpretation': self._interpretacao(
                classe,
                motivo,
                modalidades,
                concordancia,
                conflito
            ),
            'generated_at_utc': datetime.now(timezone.utc).isoformat()
        }
        arquivo_json = pasta / 'final_multimodal_report.json'
        arquivo_txt = pasta / 'final_multimodal_report.txt'
        resultado.update(json_path=str(arquivo_json), txt_path=str(arquivo_txt))
        arquivo_json.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=4, default=str),
            encoding='utf-8'
        )
        arquivo_txt.write_text(self._relatorio_txt(resultado), encoding='utf-8')
        print(f"\nFUSÃO MULTIMODAL: {classe} | score={(score if score is not None else 'N/A')}")
        print(f'Relatórios: {arquivo_json} | {arquivo_txt}')
        return resultado

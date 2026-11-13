# Reúne os detectores usados na análise visual.
from __future__ import annotations
import cv2
from pathlib import Path


# Localiza os rostos e guarda os recortes encontrados.
class FaceDetector:

    def __init__(
        self,
        model_path='models/visual/face_detection_yunet_2023mar.onnx',
        output_dir='output/faces',
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            raise FileNotFoundError(f'Modelo YuNet não encontrado: {self.model_path}')
        self.detector = cv2.FaceDetectorYN_create(
            str(self.model_path),
            '',
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )

    def _clear_output(self):
        for pattern in ['frame_*_face_*.jpg', 'detected_*.jpg']:
            for file_path in self.output_dir.glob(pattern):
                file_path.unlink(missing_ok=True)

    def detect_faces(self, frame_data):
        self._clear_output()
        detected_faces = []
        total_faces = 0
        for frame_info in frame_data['frames']:
            frame_path = Path(frame_info['path'])
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f'Não foi possível abrir: {frame_path}')
                continue
            height, width = image.shape[:2]
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(image)
            frame_faces = []
            if faces is not None:
                for face_index, face in enumerate(faces):
                    x = int(face[0])
                    y = int(face[1])
                    face_width = int(face[2])
                    face_height = int(face[3])
                    confidence = float(face[-1])
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(width, x + face_width)
                    y2 = min(height, y + face_height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    face_crop = image[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    face_filename = self.output_dir / f"frame_{frame_info['frame_index']:06d}_face_{face_index:02d}.jpg"
                    cv2.imwrite(str(face_filename), face_crop)
                    face_result = {
                        'frame_index': frame_info['frame_index'],
                        'timestamp': frame_info['timestamp'],
                        'frame_path': frame_path,
                        'face_path': face_filename,
                        'confidence': confidence,
                        'bounding_box': {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
                    }
                    detected_faces.append(face_result)
                    frame_faces.append(face_result)
                    total_faces += 1
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f'{confidence:.2f}',
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            annotated_path = self.output_dir / f"detected_{frame_info['frame_index']:06d}.jpg"
            cv2.imwrite(str(annotated_path), image)
            print(f"Frame {frame_info['frame_index']:06d} | {frame_info['timestamp']:.2f}s | Rostos: {len(frame_faces)}")
        print(f'\nTotal de rostos detectados: {total_faces}')
        return {'faces': detected_faces, 'total_faces': total_faces}

    def process(self, frame_data):
        return self.detect_faces(frame_data)


import json
import re
import time
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import cv2
import numpy as np

PLATAFORMAS = {
    'SORA': ('SORA 2', 'SORA'),
    'RUNWAY': ('RUNWAY GEN 4', 'RUNWAY GEN 3', 'RUNWAYML', 'RUNWAY'),
    'GOOGLE VEO': ('GOOGLE VEO', 'VEO 3', 'VEO 2', 'VEO'),
    'KLING AI': ('KLING AI', 'KLING'),
    'PIKA': ('PIKA LABS', 'PIKA AI', 'PIKALABS', 'PIKA'),
    'HAILUO AI': ('HAILUO AI', 'HAILUO', 'MINIMAX VIDEO'),
    'LUMA DREAM MACHINE': ('LUMA DREAM MACHINE', 'DREAM MACHINE', 'LUMA AI'),
    'PIXVERSE': ('PIXVERSE AI', 'PIXVERSE'),
    'VIDU AI': ('VIDU AI', 'VIDU STUDIO'),
    'DREAMINA': ('DREAMINA AI', 'DREAMINA'),
    'SEEDANCE': ('SEEDANCE',),
    'JIMENG AI': ('JIMENG AI',),
    'ADOBE FIREFLY': ('ADOBE FIREFLY', 'FIREFLY VIDEO'),
    'SYNTHESIA': ('SYNTHESIA',),
    'HEYGEN': ('HEYGEN',),
    'D-ID': ('D ID', 'D-ID'),
    'DEEPBRAIN AI': ('DEEPBRAIN AI', 'AI STUDIOS'),
    'COLOSSYAN': ('COLOSSYAN',),
    'ELAI': ('ELAI IO', 'ELAI.IO'),
    'HOUR ONE': ('HOUR ONE AI',),
    'VIDNOZ AI': ('VIDNOZ AI', 'VIDNOZ'),
    'VIRBO': ('WONDERSHARE VIRBO', 'VIRBO'),
    'FLIKI': ('FLIKI AI', 'FLIKI'),
    'PICTORY': ('PICTORY AI', 'PICTORY'),
    'VEED': ('VEED.IO', 'VEED AI'),
    'KAIBER': ('KAIBER AI', 'KAIBER'),
    'GENMO': ('GENMO AI', 'GENMO'),
    'HIGGSFIELD': ('HIGGSFIELD AI', 'HIGGSFIELD'),
    'POLLO AI': ('POLLO AI',),
    'HEDRA': ('HEDRA AI', 'HEDRA'),
    'HAIPER': ('HAIPER AI', 'HAIPER'),
    'VIGGLE': ('VIGGLE AI', 'VIGGLE'),
    'LTX STUDIO': ('LTX STUDIO', 'LTX VIDEO', 'LTX 2'),
    'LEONARDO AI': ('LEONARDO AI',),
    'KREA': ('KREA AI',),
    'HUNYUANVIDEO': ('HUNYUANVIDEO', 'HUNYUAN VIDEO'),
    'COGVIDEOX': ('COGVIDEOX',),
    'OPEN SORA': ('OPEN SORA', 'OPEN-SORA'),
    'WAN AI': ('WAN AI', 'WAN VIDEO'),
    'MOTION VIDEO': ('MOTION VIDEO AI',),
    'VYOND': ('VYOND GO', 'VYOND'),
    'AKOOL': ('AKOOL AI', 'AKOOL'),
    'MAGIC HOUR': ('MAGIC HOUR AI',),
    'CLIPFLY': ('CLIPFLY',),
    'DEEVID': ('DEEVID AI',),
    'CLIPZAP': ('CLIPZAP AI',),
    'RENDERFOREST': ('RENDERFOREST AI',),
    'ANIMAKER': ('ANIMAKER AI',),
    'DECOHERE': ('DECOHERE',),
    'NEURAL FRAMES': ('NEURAL FRAMES',),
    'DEEPAI': ('DEEPAI VIDEO',),
    'DZINE AI': ('DZINE AI',),
    'SEAART AI': ('SEAART AI',),
    'DOMOAI': ('DOMOAI',),
    'GOENHANCE': ('GOENHANCE AI',),
    'GROK IMAGINE': ('GROK IMAGINE',),
    'TIKTOK SYMPHONY': ('TIKTOK SYMPHONY', 'SYMPHONY CREATIVE STUDIO'),
    'CAPTIONS AI': ('CAPTIONS AI',),
    'JOGGAI': ('JOGGAI',),
    'ARCADS AI': ('ARCADS AI',),
    'CREATIFY AI': ('CREATIFY AI', 'CREATIFY'),
    'TAVUS': ('TAVUS AI',),
    'PIPIO': ('PIPIO AI',),
    'YEPIC': ('YEPIC AI',),
    'SYNTHESYS': ('SYNTHESYS AI',),
    'MAGICLIGHT': ('MAGICLIGHT AI',),
    'TOPVIEW': ('TOPVIEW AI',),
    'WAV2LIP': ('WAV2LIP',),
    'VIDEORETALKING': ('VIDEORETALKING',),
    'FACEFUSION': ('FACEFUSION',),
    'REFACE': ('REFACE AI',)
}
PLATAFORMAS_SOMENTE_COM_CONTEXTO = {
    'CANVA',
    'CAPCUT',
    'INVIDEO',
    'FLEXCLIP',
    'LUMEN5',
    'OPENAI',
    'GOOGLE',
    'MICROSOFT',
    'META AI',
    'FREEPIK',
    'TIKTOK',
    'VEED',
    'DEEPFAKELAB',
    'FACE SWAP',
    'SADTALKER',
    'LUMIERE',
    'ANIMATEDIFF'
}
FRASES_GERACAO = (
    'AI GENERATED',
    'AI GENERATED VIDEO',
    'GENERATED BY AI',
    'AIGENERATED',
    'AI GENERATION',
    'AI GENERATIVE',
    'GENERATED WITH AI',
    'GENERATED USING AI',
    'CREATED WITH AI',
    'CREATED BY AI',
    'MADE WITH AI',
    'GENERATIVE AI',
    'GENERATIVE IA',
    'GENERATED BY ARTIFICIAL INTELLIGENCE',
    'SYNTHETIC MEDIA',
    'ARTIFICIALLY GENERATED VIDEO',
    'ARTIFICIAL INTELLIGENCE GENERATED',
    'SYNTHETIC VIDEO',
    'AI SYNTHESIZED',
    'AI CREATED',
    'GERADO POR IA',
    'GERADO COM IA',
    'FEITO COM IA',
    'CRIADO POR IA',
    'CONTEUDO GERADO POR IA',
    'VIDEO GERADO POR IA',
    'GERADO POR INTELIGENCIA ARTIFICIAL',
    'INTELIGENCIA ARTIFICIAL GEROU'
)
EXPRESSAO_ATRIBUICAO = re.compile('\\b(?:MADE|CREATED|GENERATED|POWERED|PRODUCED|BUILT|EDITED)\\s+(?:BY|WITH|USING)\\s+|\\b(?:GERADO|CRIADO|FEITO|PRODUZIDO)\\s+(?:POR|COM)\\s+')


def normalizar(texto):
    sem_acento = ''.join((letra for letra in unicodedata.normalize('NFKD', str(texto or '')) if not unicodedata.combining(letra)))
    return re.sub('[^A-Z0-9]+', ' ', sem_acento.upper()).strip()


def corresponde_frase(texto, termo):
    texto, termo = (normalizar(texto), normalizar(termo))
    return bool(termo and re.search('(?<![A-Z0-9])' + re.escape(termo) + '(?![A-Z0-9])', texto))


# Executa o reconhecimento de texto nos recortes selecionados.
def reconhecer_texto(texto):
    texto = normalizar(texto)
    if not texto:
        return None
    texto = re.sub('\\b(?:AL|A1|A I)\\s+GENERAT(?:ED|EO|E0|D)\\b', 'AI GENERATED', texto)
    texto = re.sub('\\bAI\\s+GENERAT\\s+ED\\b', 'AI GENERATED', texto)
    texto = re.sub('\\bAI\\s+GENERATEO\\b', 'AI GENERATED', texto)
    texto = re.sub('\\bAIGENERATED\\b', 'AI GENERATED', texto)
    if not texto:
        return None
    for termo in FRASES_GERACAO:
        if corresponde_frase(texto, termo):
            return ('DECLARACAO_IA', termo, 'EXPLICITA')
    for marca, variantes in PLATAFORMAS.items():
        for variante in sorted(variantes, key=len, reverse=True):
            if corresponde_frase(texto, variante):
                if EXPRESSAO_ATRIBUICAO.search(texto):
                    return ('ATRIBUICAO_PLATAFORMA', marca, 'EXPLICITA')
                return ('MARCA_DE_PLATAFORMA', marca, 'CONTEXTUAL')
    if EXPRESSAO_ATRIBUICAO.search(texto):
        for termo in PLATAFORMAS_SOMENTE_COM_CONTEXTO:
            if corresponde_frase(texto, termo):
                return ('ATRIBUICAO_PLATAFORMA', termo, 'EXPLICITA')
    return None


# Procura marcas textuais e reúne as evidências para confirmação.
class DetectorMarcasDaguaVisiveis:
    ORIENTACOES = ('NORMAL', 'ESPELHADO_HORIZONTAL', 'ESPELHADO_VERTICAL', 'ROTACIONADO_180')

    def __init__(
        self,
        quadros_maximos=5,
        largura_maxima=1440,
        confianca_ocr_minima=0.78,
        evidencias_minimas=2,
        separar_segundos=0.5,
        ocr=None,
        quadros_adaptativos=24,
        limite_chamadas_ocr=150,
        tempo_limite_segundos=38.0,
        limite_frames_longo=72,
        intervalo_longo_segundos=8.0,
        composicao_temporal=True,
        minimo_temporal=True
    ):
        self.quadros_maximos = max(2, int(quadros_maximos))
        self.largura_maxima = max(640, int(largura_maxima))
        self.confianca_ocr_minima = float(confianca_ocr_minima)
        self.evidencias_minimas = max(2, int(evidencias_minimas))
        self.separar_segundos = float(separar_segundos)
        self.quadros_adaptativos = max(0, int(quadros_adaptativos))
        self.limite_chamadas_ocr = max(4, int(limite_chamadas_ocr))
        self.tempo_limite_segundos = max(1.0, float(tempo_limite_segundos))
        self.limite_frames_longo = max(8, int(limite_frames_longo))
        self.intervalo_longo_segundos = max(2.0, float(intervalo_longo_segundos))
        self.composicao_temporal = bool(composicao_temporal)
        self.minimo_temporal = bool(minimo_temporal)
        self._estado = None
        if ocr is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as error:
                raise RuntimeError("Triagem de marca d'água precisa de RapidOCR. Execute: python -m pip install rapidocr onnxruntime") from error
            diretorio_ocr = Path('models/visual/ocr')
            diretorio_ocr.mkdir(parents=True, exist_ok=True)
            ocr = RapidOCR(params={'Global.model_root_dir': str(diretorio_ocr.resolve())})
        self.ocr = ocr

    @staticmethod
    def _orientar(frame, orientacao):
        if orientacao == 'ESPELHADO_HORIZONTAL':
            return cv2.flip(frame, 1)
        if orientacao == 'ESPELHADO_VERTICAL':
            return cv2.flip(frame, 0)
        if orientacao == 'ROTACIONADO_180':
            return cv2.flip(frame, -1)
        return frame

    @staticmethod
    def _desorientar_bbox(points, orientacao, w, h):
        pts = np.asarray(points, dtype=float).reshape(-1, 2).copy()
        if orientacao in ('ESPELHADO_HORIZONTAL', 'ROTACIONADO_180'):
            pts[:, 0] = w - 1 - pts[:, 0]
        if orientacao in ('ESPELHADO_VERTICAL', 'ROTACIONADO_180'):
            pts[:, 1] = h - 1 - pts[:, 1]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return [
            round(max(0.0, x1), 1),
            round(max(0.0, y1), 1),
            round(min(float(w), x2), 1),
            round(min(float(h), y2), 1)
        ]

    @staticmethod
    def _area_video_ativa(frame):
        altura, largura = frame.shape[:2]
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        amostra = cinza[:, ::max(1, largura // 200)]
        atividade = np.mean(amostra > 18, axis=1) > 0.19
        segmentos, inicio = ([], None)
        for i, ativo in enumerate(list(atividade) + [False]):
            if ativo and inicio is None:
                inicio = i
            elif not ativo and inicio is not None:
                if i - inicio >= altura * 0.12:
                    segmentos.append((inicio, i))
                inicio = None
        if not segmentos:
            return (0, 0, largura, altura)
        superior, inferior = max(segmentos, key=lambda segmento: segmento[1] - segmento[0])
        if superior <= altura * 0.1 and inferior >= altura * 0.9:
            return (0, 0, largura, altura)
        if inferior - superior >= altura * 0.97 or inferior - superior < altura * 0.18:
            return (0, 0, largura, altura)
        return (0, int(superior), largura, int(inferior))

    @staticmethod
    def _regiao(bbox, largura, altura, area_ativa=None):
        x1, y1, x2, y2 = bbox
        esquerda, topo, direita, fundo = area_ativa or (0, 0, largura, altura)
        if not (esquerda <= (x1 + x2) / 2 <= direita and topo <= (y1 + y2) / 2 <= fundo):
            esquerda, topo, direita, fundo = (0, 0, largura, altura)
        cx = ((x1 + x2) / 2 - esquerda) / max(1, direita - esquerda)
        cy = ((y1 + y2) / 2 - topo) / max(1, fundo - topo)
        if cx < 0.3 and cy < 0.3:
            return 'SUPERIOR_ESQUERDA'
        if cx > 0.7 and cy < 0.3:
            return 'SUPERIOR_DIREITA'
        if cy < 0.2:
            return 'SUPERIOR_CENTRO'
        if cx < 0.3 and cy > 0.7:
            return 'INFERIOR_ESQUERDA'
        if cx > 0.7 and cy > 0.7:
            return 'INFERIOR_DIREITA'
        if cy > 0.82:
            return 'INFERIOR_CENTRO'
        return 'INTERIOR'

    @staticmethod
    def _faixas_para_ocr(frame):
        h, w = frame.shape[:2]
        faixas = [
            ('BORDA_INFERIOR', (0, int(h * 0.75), w, h)),
            ('BORDA_SUPERIOR', (0, 0, w, max(1, int(h * 0.25))))
        ]
        x1, y1, x2, y2 = DetectorMarcasDaguaVisiveis._area_video_ativa(frame)
        if y1 > h * 0.04 or y2 < h * 0.96:
            altura_ativa = y2 - y1
            faixas.extend([
                ('CONTEUDO_INFERIOR', (x1, max(y1, int(y2 - altura_ativa * 0.34)), x2, y2)),
                ('CONTEUDO_SUPERIOR', (x1, y1, x2, min(y2, int(y1 + altura_ativa * 0.34))))
            ])
        faixas.extend([
            ('LATERAL_ESQUERDA', (0, int(h * 0.25), max(1, int(w * 0.24)), int(h * 0.75))),
            ('LATERAL_DIREITA', (int(w * 0.76), int(h * 0.25), w, int(h * 0.75)))
        ])
        return [(nome, roi) for nome, roi in faixas if roi[2] - roi[0] > 8 and roi[3] - roi[1] > 8]

    @staticmethod
    def _grupos_texto(candidatos, w, h):
        saida = list(candidatos)
        ordenados = sorted(
            candidatos,
            key=lambda t: (float(t[2][:, 1].mean()), float(t[2][:, 0].mean()))
        )
        for i, a in enumerate(ordenados):
            for b in ordenados[i + 1:i + 5]:
                ya, yb = (a[2][:, 1].mean(), b[2][:, 1].mean())
                ha, hb = (np.ptp(a[2][:, 1]), np.ptp(b[2][:, 1]))
                if abs(ya - yb) > max(10, ha, hb) * 0.9:
                    continue
                dx = b[2][:, 0].min() - a[2][:, 0].max()
                if dx < -max(ha, hb) * 0.7 or dx > max(24, 2.5 * max(ha, hb)):
                    continue
                x1 = min(a[2][:, 0].min(), b[2][:, 0].min())
                y1 = min(a[2][:, 1].min(), b[2][:, 1].min())
                x2 = max(a[2][:, 0].max(), b[2][:, 0].max())
                y2 = max(a[2][:, 1].max(), b[2][:, 1].max())
                if x2 - x1 > w * 0.5 or y2 - y1 > h * 0.18:
                    continue
                saida.append((
                    a[0] + ' ' + b[0],
                    min(a[1], b[1]),
                    np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
                ))
        return saida

    @staticmethod
    def _preparar_recorte(recorte, fator_desejado=4.0, realcar=True):
        h, w = recorte.shape[:2]
        fator = min(5.0, max(1.0, float(fator_desejado)))
        fator = min(fator, 1100 / max(1, w), 650 / max(1, h))
        fator = max(1.0, fator)
        ampliada = cv2.resize(recorte, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)
        if realcar:
            lab = cv2.cvtColor(ampliada, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
            ampliada = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        return (ampliada, fator)

    @staticmethod
    def _rois_prioritarias(frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = DetectorMarcasDaguaVisiveis._area_video_ativa(frame)
        cw, ch = (x2 - x1, y2 - y1)
        rw, rh = (max(24, int(cw * 0.31)), max(20, int(ch * 0.25)))
        zonas = [
            ('CONTEUDO_INFERIOR', 'DIREITA', (x2 - rw, y2 - rh, x2, y2)),
            ('CONTEUDO_INFERIOR', 'ESQUERDA', (x1, y2 - rh, x1 + rw, y2))
        ]
        if y1 > h * 0.12 or y2 < h * 0.88:
            rh_tela = max(22, int(h * 0.16))
            zonas += [
                ('TELA_INFERIOR', 'ESQUERDA', (0, h - rh_tela, min(w, int(w * 0.32)), h)),
                ('TELA_INFERIOR', 'DIREITA', (max(0, int(w * 0.68)), h - rh_tela, w, h))
            ]
        zonas += [
            ('CONTEUDO_SUPERIOR', 'DIREITA', (x2 - rw, y1, x2, y1 + rh)),
            ('CONTEUDO_SUPERIOR', 'ESQUERDA', (x1, y1, x1 + rw, y1 + rh))
        ]
        return [
            (nome, lado, tuple(map(int, caixa))) for nome,
            lado,
            caixa in zonas if caixa[2] - caixa[0] > 12 and caixa[3] - caixa[1] > 10
        ]

    @staticmethod
    def _reconhecer_texto_tolerante(texto):
        padrao = reconhecer_texto(texto)
        if padrao:
            return padrao
        t = normalizar(texto)
        match = re.search('(?<![A-Z0-9])(?:AI|AL|A1|A I)\\s*([A-Z0-9\\s]{6,14})', t)
        if not match:
            return None
        cauda = re.sub('\\s+', '', match.group(1))
        cauda = cauda[:9]
        if len(cauda) >= 8 and SequenceMatcher(None, cauda, 'GENERATED').ratio() >= 0.83:
            return ('DECLARACAO_IA', 'AI GENERATED', 'EXPLICITA')
        return None

    @staticmethod
    def _rois_micro(frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = DetectorMarcasDaguaVisiveis._area_video_ativa(frame)
        cw, ch = (x2 - x1, y2 - y1)
        rw, rh = (max(30, int(cw * 0.25)), max(20, int(ch * 0.17)))
        dados = [
            ('CONTEUDO_INFERIOR', 'DIREITA', (x2 - rw, y2 - rh, x2, y2)),
            ('CONTEUDO_INFERIOR', 'ESQUERDA', (x1, y2 - rh, x1 + rw, y2))
        ]
        if y1 > h * 0.12 or y2 < h * 0.88:
            tw, th = (max(30, int(w * 0.27)), max(20, int(h * 0.13)))
            dados.extend([
                ('TELA_INFERIOR', 'DIREITA', (w - tw, h - th, w, h)),
                ('TELA_INFERIOR', 'ESQUERDA', (0, h - th, tw, h))
            ])
        return [
            (nome, lado, tuple((int(v) for v in caixa))) for nome,
            lado,
            caixa in dados if caixa[2] > caixa[0] and caixa[3] > caixa[1]
        ]

    @staticmethod
    def _preparar_variante_recorte(recorte, variante):
        if variante == 'NATIVO':
            return (recorte.copy(), 1.0)
        if variante == 'CLAHE_2X':
            lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
            melhor = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            h, w = melhor.shape[:2]
            fator = min(2.5, 960 / max(1, w), 480 / max(1, h))
            fator = max(1.0, fator)
            return (
                cv2.resize(melhor, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC),
                fator
            )
        if variante == 'BINARIO':
            cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            cinza = cv2.GaussianBlur(cinza, (3, 3), 0)
            binaria = cv2.adaptiveThreshold(
                cinza,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                17,
                5
            )
            h, w = binaria.shape[:2]
            fator = max(1.0, min(2.0, 960 / max(1, w), 480 / max(1, h)))
            binaria = cv2.resize(binaria, None, fx=fator, fy=fator, interpolation=cv2.INTER_NEAREST)
            return (cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR), fator)
        raise ValueError('Variante desconhecida: ' + str(variante))

    def _orcamento_disponivel(self):
        e = self._estado
        if e['chamadas_ocr'] >= self.limite_chamadas_ocr:
            e['limite_atingido'] = 'LIMITE_CHAMADAS_OCR'
            return False
        if time.perf_counter() - e['inicio_scan'] >= e.get(
            'tempo_limite_dinamico',
            self.tempo_limite_segundos
        ):
            e['limite_atingido'] = 'LIMITE_TEMPO_OCR'
            return False
        return True

    def _extrair_evidencias(self, frame, timestamp, indice, modo='INICIAL'):
        e = self._estado
        h, w = frame.shape[:2]
        area_ativa = self._area_video_ativa(frame)
        evidencias, vistos = ([], {})
        registrar_antes = len(e['textos_ocr_amostrados'])

        def consultar(imagem, orientacao, nome, caixa, fator=1.0, escala=1.0, variante='NATIVO'):
            if not self._orcamento_disponivel():
                return False
            x1, y1, x2, y2 = caixa
            try:
                saida = self.ocr(imagem)
                e['chamadas_ocr'] += 1
                textos = list(getattr(saida, 'txts', None) or [])
                scores = list(getattr(saida, 'scores', None) or [])
                boxes = getattr(saida, 'boxes', None)
            except Exception as erro:
                e['chamadas_ocr'] += 1
                e['erros_ocr'].append({
                    'frame_index': int(indice),
                    'origem_busca': nome,
                    'erro': f'{type(erro).__name__}: {erro}'
                })
                return True
            if boxes is None:
                return True
            candidatos = []
            for i, txt in enumerate(textos):
                if i >= len(scores) or i >= len(boxes):
                    continue
                try:
                    conf = float(scores[i])
                    pontos = np.asarray(boxes[i], dtype=float).reshape(-1, 2)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(conf) or not np.isfinite(pontos).all() or len(pontos) < 3:
                    continue
                if len(e['textos_ocr_amostrados']) < 420 and conf >= 0.32:
                    e['textos_ocr_amostrados'].append({
                        'frame_index': int(indice),
                        'timestamp': round(float(timestamp), 3),
                        'texto': str(txt)[:110],
                        'confianca': round(conf, 4),
                        'origem_busca': nome,
                        'orientacao': orientacao,
                        'variante': variante
                    })
                if conf >= 0.5:
                    candidatos.append((str(txt), conf, pontos))
            for txt, conf, pontos in self._grupos_texto(
                candidatos,
                imagem.shape[1],
                imagem.shape[0]
            ):
                rotulo = self._reconhecer_texto_tolerante(txt)
                if rotulo is None:
                    continue
                categoria, marca, forca = rotulo
                norm = normalizar(txt)
                if categoria == 'DECLARACAO_IA':
                    minimo = 0.73 if nome == 'FRAME_INTEIRO' else 0.67
                elif marca == 'GOOGLE VEO' and norm == 'VEO':
                    minimo = 0.81
                else:
                    minimo = 0.78 if nome == 'FRAME_INTEIRO' else 0.74
                if conf < minimo:
                    continue
                coords = pontos.copy() / float(fator)
                coords[:, 0] = coords[:, 0] / float(escala) + x1
                coords[:, 1] = coords[:, 1] / float(escala) + y1
                bbox = self._desorientar_bbox(coords, orientacao, w, h)
                regiao = self._regiao(bbox, w, h, area_ativa)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(1, w * h)
                compativel = bool(regiao != 'INTERIOR' and area <= 0.075)
                cx = (bbox[0] + bbox[2]) / (2 * w)
                cy = (bbox[1] + bbox[3]) / (2 * h)
                item = {
                    'marca': marca,
                    'categoria': categoria,
                    'forca': forca,
                    'texto_lido': txt,
                    'confianca_ocr': round(conf, 4),
                    'timestamp': round(float(timestamp), 3),
                    'frame_index': int(indice),
                    'regiao': regiao,
                    'bbox_xyxy': bbox,
                    'orientacao_corrigida': orientacao,
                    'posicao_compativel': compativel,
                    'centro_normalizado': [round(cx, 4), round(cy, 4)],
                    'origem_busca': nome,
                    'subregiao': nome,
                    'variante': variante,
                    'frame_composto': bool(modo in ('COMPOSTO', 'MINIMO_TEMPORAL'))
                }
                chave = (marca, regiao)
                if chave not in vistos or conf > evidencias[vistos[chave]]['confianca_ocr']:
                    if chave in vistos:
                        evidencias[vistos[chave]] = item
                    else:
                        vistos[chave] = len(evidencias)
                        evidencias.append(item)
            return True

        def roi(orientacao, nome, caixa, variante='NATIVO'):
            orientada = self._orientar(frame, orientacao)
            x1, y1, x2, y2 = caixa
            recorte = orientada[y1:y2, x1:x2]
            if recorte.size == 0:
                return False
            img, fator = self._preparar_variante_recorte(recorte, variante)
            return consultar(img, orientacao, nome, caixa, fator, 1.0, variante)
        if modo == 'INICIAL':
            escala = min(1.0, self.largura_maxima / max(1, w))
            img = cv2.resize(
                frame,
                (max(1, int(w * escala)), max(1, int(h * escala))),
                interpolation=cv2.INTER_AREA
            ) if escala < 1 else frame
            consultar(img, 'NORMAL', 'FRAME_INTEIRO', (0, 0, w, h), 1.0, escala, 'REDUZIDO')
            if not any((t['posicao_compativel'] for t in evidencias)) and self._orcamento_disponivel():
                consultar(
                    cv2.flip(img, 1),
                    'ESPELHADO_HORIZONTAL',
                    'FRAME_INTEIRO',
                    (0, 0, w, h),
                    1.0,
                    escala,
                    'REDUZIDO'
                )
        micros = self._rois_micro(frame)
        prioridade = micros[:4] if modo == 'INICIAL' else micros[:2]
        for nome, lado, caixa in prioridade:
            if not self._orcamento_disponivel():
                break
            roi('NORMAL', nome, caixa, 'CLAHE_2X')
            if any((t['posicao_compativel'] for t in evidencias)):
                break
        if modo == 'MINIMO_TEMPORAL' and (not evidencias) and self._orcamento_disponivel():
            for nome, lado, caixa in micros[:2]:
                if not self._orcamento_disponivel():
                    break
                roi('NORMAL', nome, caixa, 'NATIVO')
                if any((t['posicao_compativel'] for t in evidencias)):
                    break
        if not any((t['posicao_compativel'] for t in evidencias)) and self._orcamento_disponivel():
            if modo == 'INICIAL' or (modo == 'ADAPTATIVO' and indice % 3 == 0):
                img = self._orientar(frame, 'ESPELHADO_HORIZONTAL')
                for nome, lado, caixa in self._rois_micro(img)[:2]:
                    if not self._orcamento_disponivel():
                        break
                    roi('ESPELHADO_HORIZONTAL', nome, caixa, 'CLAHE_2X')
                    if any((t['posicao_compativel'] for t in evidencias)):
                        break
        amostras = e['textos_ocr_amostrados'][registrar_antes:]
        vestigio = any((re.search(
            '(?:\\bAI\\b|\\bAL\\b|\\bVEO\\b|GENERAT|SORA|KLING)',
            normalizar(t['texto'])
        ) for t in amostras))
        if not evidencias and self._orcamento_disponivel() and (vestigio or indice % 4 == 0 or modo == 'MINIMO_TEMPORAL'):
            for nome, lado, caixa in micros[:2]:
                if not self._orcamento_disponivel():
                    break
                roi('NORMAL', nome, caixa, 'BINARIO')
                if any((t['posicao_compativel'] for t in evidencias)):
                    break
        if modo == 'INICIAL' and (not evidencias) and self._orcamento_disponivel() and (indice % 5 == 0):
            img = cv2.rotate(frame, cv2.ROTATE_180)
            for nome, lado, caixa in self._rois_micro(img)[:2]:
                if not self._orcamento_disponivel():
                    break
                roi('ROTACIONADO_180', nome, caixa, 'CLAHE_2X')
                if any((t['posicao_compativel'] for t in evidencias)):
                    break
        return evidencias

    def _indices_temporais(self, total, fps, iniciais, candidatos):
        duracao = total / max(0.001, fps)
        adicionais = max(
            self.quadros_adaptativos,
            int(np.ceil(duracao / self.intervalo_longo_segundos))
        )
        adicionais = min(self.limite_frames_longo, adicionais, max(0, total - len(iniciais)))
        ocupados = set(iniciais)
        propostos = []
        fronteiras = [0, *sorted(ocupados), total - 1]
        intervalos = sorted(set(fronteiras))
        enquanto = []
        for a, b in zip(intervalos, intervalos[1:]):
            enquanto.append((a, b))
        for _ in range(adicionais):
            enquanto.sort(key=lambda par: par[1] - par[0], reverse=True)
            if not enquanto:
                break
            a, b = enquanto.pop(0)
            if b - a < 2:
                break
            meio = (a + b) // 2
            if meio not in ocupados:
                propostos.append(meio)
                ocupados.add(meio)
            enquanto.extend([(a, meio), (meio, b)])
        vizinhos = []
        for c in sorted(candidatos, key=lambda x: -x['confianca_ocr']):
            base = c['frame_index']
            for delta in (round(fps * 0.6), round(-fps * 0.6), round(fps * 1.4), round(-fps * 1.4)):
                indice = max(0, min(total - 1, base + int(delta)))
                if indice not in ocupados and indice not in vizinhos:
                    vizinhos.append(indice)
        k = min(len(vizinhos), adicionais // 4)
        if k:
            propostos = propostos[:max(0, adicionais - k)] + vizinhos[:k]
        return propostos[:adicionais]

    def _confirmar(self, evidencias):
        grupos = defaultdict(list)
        for item in evidencias:
            if item['posicao_compativel']:
                grupos[item['marca'], item['regiao']].append(item)
        for (_, _), itens in sorted(grupos.items(), key=lambda par: -len(par[1])):
            unicos = {}
            for item in itens:
                idx = item['frame_index']
                if idx not in unicos or item['confianca_ocr'] > unicos[idx]['confianca_ocr']:
                    unicos[idx] = item
            sequencia = sorted(unicos.values(), key=lambda t: t['timestamp'])
            if len(sequencia) < self.evidencias_minimas:
                continue
            minimo = self.separar_segundos
            if any((item.get('frame_composto') for item in sequencia)):
                minimo = max(minimo, 1.25)
            if sequencia[-1]['timestamp'] - sequencia[0]['timestamp'] < minimo:
                continue
            xs = [t['centro_normalizado'][0] for t in sequencia]
            ys = [t['centro_normalizado'][1] for t in sequencia]
            if max(xs) - min(xs) <= 0.11 and max(ys) - min(ys) <= 0.11:
                return ('MESMA_MARCA_PERSISTENTE', sequencia)
        explicitas = [x for x in evidencias if x['posicao_compativel'] and x['categoria'] == 'DECLARACAO_IA' and (x['confianca_ocr'] >= 0.79)]
        marcas = [x for x in evidencias if x['posicao_compativel'] and x['categoria'] in (
            'MARCA_DE_PLATAFORMA',
            'ATRIBUICAO_PLATAFORMA'
        ) and (x['confianca_ocr'] >= 0.83)]
        for e in sorted(explicitas, key=lambda a: -a['confianca_ocr']):
            for p in sorted(marcas, key=lambda a: -a['confianca_ocr']):
                if e['frame_index'] == p['frame_index'] or abs(e['timestamp'] - p['timestamp']) < self.separar_segundos:
                    continue
                return ('PISTAS_COMPLEMENTARES_EM_FRAMES_DISTINTOS', [e, p])
        return (None, None)

    @staticmethod
    def _escrever_imagem(caminho, imagem, extensao):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        parametros = [cv2.IMWRITE_JPEG_QUALITY, 93] if extensao == '.jpg' else []
        sucesso, dados = cv2.imencode(extensao, imagem, parametros)
        if not sucesso:
            raise OSError(f'OpenCV não conseguiu codificar {caminho.name}')
        caminho.write_bytes(dados.tobytes())
        if not caminho.is_file() or caminho.stat().st_size == 0:
            raise OSError(f'Arquivo de imagem não foi gravado: {caminho}')

    @staticmethod
    def _limpar_arquivos_visuais_anteriores(diretorio_saida):
        raiz = Path(diretorio_saida)
        for pasta, padrao in (
            (raiz / 'evidencias_marca_dagua', 'marca_*.png'),
            (raiz / 'frames', 'frame_*.jpg')
        ):
            if pasta.is_dir():
                for arquivo in pasta.glob(padrao):
                    arquivo.unlink()

    @staticmethod
    def _guardar_evidencias_visuais(caminho_video, confirmadas, diretorio_saida, compostas=None):
        raiz = Path(diretorio_saida)
        pasta_frames = raiz / 'frames'
        pasta_evidencias = raiz / 'evidencias_marca_dagua'
        pasta_frames.mkdir(parents=True, exist_ok=True)
        pasta_evidencias.mkdir(parents=True, exist_ok=True)
        compostas = compostas or {}
        frames_salvos = []
        indices_salvos = set()
        avisos = []
        captura = cv2.VideoCapture(str(caminho_video))
        try:
            if not captura.isOpened():
                return {
                    'frames': [],
                    'avisos': [f'Não foi possível reabrir o vídeo: {caminho_video}'],
                    'pasta_frames': str(pasta_frames),
                    'pasta_evidencias': str(pasta_evidencias)
                }
            for posicao, item in enumerate(confirmadas):
                indice = int(item['frame_index'])
                captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
                sucesso, original = captura.read()
                if not sucesso or original is None:
                    aviso = f'Não foi possível reler o frame original {indice}.'
                    avisos.append(aviso)
                    item['erro_salvamento'] = aviso
                    continue
                caminho_frame = pasta_frames / f'frame_{indice:06d}.jpg'
                if indice not in indices_salvos:
                    try:
                        DetectorMarcasDaguaVisiveis._escrever_imagem(caminho_frame, original, '.jpg')
                        frames_salvos.append({
                            'frame_index': indice,
                            'timestamp': item.get('timestamp'),
                            'path': str(caminho_frame),
                            'origem': 'OCR_MARCA_DAGUA'
                        })
                        indices_salvos.add(indice)
                    except (OSError, cv2.error) as erro:
                        aviso = f'Falha ao salvar frame {indice}: {erro}'
                        avisos.append(aviso)
                        item['erro_salvamento'] = aviso
                if caminho_frame.is_file():
                    item['frame_original'] = str(caminho_frame)
                composicao = compostas.get(indice) if item.get('frame_composto') else None
                imagem = composicao.copy() if composicao is not None else original.copy()
                x1, y1, x2, y2 = [int(round(v)) for v in item['bbox_xyxy']]
                altura, largura = imagem.shape[:2]
                x1, x2 = sorted((max(0, min(largura - 1, x1)), max(0, min(largura - 1, x2))))
                y1, y2 = sorted((max(0, min(altura - 1, y1)), max(0, min(altura - 1, y2))))
                cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    imagem,
                    str(item['marca'])[:42],
                    (max(4, x1), max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                caminho_evidencia = pasta_evidencias / f'marca_{indice:08d}_{posicao}.png'
                try:
                    DetectorMarcasDaguaVisiveis._escrever_imagem(caminho_evidencia, imagem, '.png')
                    item['imagem_evidencia'] = str(caminho_evidencia)
                    item['tipo_imagem_evidencia'] = 'COMPOSICAO_TEMPORAL' if composicao is not None else 'FRAME_ORIGINAL_ANOTADO'
                except (OSError, cv2.error) as erro:
                    aviso = f'Falha ao salvar evidência do frame {indice}: {erro}'
                    avisos.append(aviso)
                    item['erro_salvamento'] = aviso
        finally:
            captura.release()
        return {
            'frames': sorted(frames_salvos, key=lambda item: item['frame_index']),
            'avisos': avisos,
            'pasta_frames': str(pasta_frames),
            'pasta_evidencias': str(pasta_evidencias)
        }

    def _compor_minimo_persistente(self, captura, indice_central, total, fps):
        deslocamentos = (-0.6, -0.3, 0.0, 0.3, 0.6)
        indices = sorted(set((max(
            0,
            min(total - 1, int(indice_central + round(d * fps)))
        ) for d in deslocamentos)))
        if len(indices) < 3:
            return (None, [])
        quadros = []
        for ind in indices:
            captura.set(cv2.CAP_PROP_POS_FRAMES, int(ind))
            ok, quadro = captura.read()
            if ok and quadro is not None:
                quadros.append(quadro)
        if len(quadros) < 3 or any((q.shape != quadros[0].shape for q in quadros)):
            return (None, [])
        central = quadros[len(quadros) // 2].copy()
        for _, _, (x1, y1, x2, y2) in self._rois_micro(central)[:2]:
            recortes = np.stack([q[y1:y2, x1:x2] for q in quadros], axis=0)
            central[y1:y2, x1:x2] = np.min(recortes, axis=0)
        return (central, indices)

    def analisar(self, caminho_video, saida_json=None):
        inicio = time.perf_counter()
        video = Path(caminho_video)
        captura = cv2.VideoCapture(str(video))
        if not captura.isOpened():
            raise RuntimeError(f'Impossível abrir vídeo para OCR: {video}')
        fps = float(captura.get(cv2.CAP_PROP_FPS) or 0)
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            captura.release()
            raise RuntimeError('FPS ou total de frames inválido na triagem OCR')
        duracao = total / fps
        avisos_organizacao = []
        if saida_json is not None:
            try:
                self._limpar_arquivos_visuais_anteriores(Path(saida_json).parent)
            except OSError as erro:
                avisos_organizacao.append(f'Não foi possível limpar imagens antigas: {erro}')
        iniciais = sorted(set((int(round(t)) for t in np.linspace(0, total - 1, self.quadros_maximos))))
        tempo_dinamico = min(
            75.0,
            self.tempo_limite_segundos + max(0.0, duracao - 120.0) / 120.0 * 7.0
        )
        e = self._estado = {
            'inicio_scan': time.perf_counter(),
            'chamadas_ocr': 0,
            'limite_atingido': None,
            'textos_ocr_amostrados': [],
            'erros_ocr': [],
            'tempo_limite_dinamico': tempo_dinamico
        }
        candidatos = []
        erros = []
        analisados = []
        confirmadas = None
        tipo_confirmacao = None
        planejados = []
        composicoes = 0
        composicoes_minimo = 0
        minimos_proveniencia = []
        imagens_compostas = {}

        def processar(indice, modo, frame_override=None):
            nonlocal confirmadas, tipo_confirmacao
            if not self._orcamento_disponivel():
                return False
            if frame_override is None:
                captura.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
                ok, frame = captura.read()
                if not ok or frame is None:
                    erros.append({'frame_index': int(indice), 'erro': 'Falha ao ler frame'})
                    return True
            else:
                frame = frame_override
            if indice not in analisados:
                analisados.append(int(indice))
            try:
                candidatos.extend(self._extrair_evidencias(frame, indice / fps, indice, modo=modo))
            except Exception as erro:
                erros.append({'frame_index': int(indice), 'erro': f'{type(erro).__name__}: {erro}'})
            tipo_confirmacao, confirmadas = self._confirmar(candidatos)
            return True
        try:
            for indice in iniciais:
                if not processar(indice, 'INICIAL') or confirmadas:
                    break
            if not confirmadas and self.minimo_temporal and self._orcamento_disponivel():
                centros = [int(round((total - 1) * f)) for f in (0.2, 0.8)]
                fontes_usadas = set()
                for centro in centros:
                    if not self._orcamento_disponivel() or confirmadas:
                        break
                    composta, origens = self._compor_minimo_persistente(captura, centro, total, fps)
                    if composta is None or len(origens) < 3:
                        continue
                    if fontes_usadas.intersection(origens):
                        continue
                    fontes_usadas.update(origens)
                    composicoes_minimo += 1
                    imagens_compostas[centro] = composta
                    minimos_proveniencia.append({'centro': centro, 'indices_origem': origens})
                    processar(centro, 'MINIMO_TEMPORAL', composta)
            if not confirmadas and self._orcamento_disponivel():
                planejados = self._indices_temporais(total, fps, iniciais, candidatos)
                for indice in planejados:
                    if not processar(indice, 'ADAPTATIVO') or confirmadas:
                        break
            if not confirmadas and self.composicao_temporal and self._orcamento_disponivel():
                bases = sorted(set(analisados))
                indices_de_interesse = [x['frame_index'] for x in sorted(
                    candidatos,
                    key=lambda d: -d['confianca_ocr']
                )]
                amostras = []
                for i in indices_de_interesse + [bases[k] for k in np.linspace(
                    0,
                    len(bases) - 1,
                    min(4, len(bases)),
                    dtype=int
                )]:
                    if i not in amostras:
                        amostras.append(i)
                    if len(amostras) >= 4:
                        break
                for indice in amostras:
                    if not self._orcamento_disponivel() or confirmadas:
                        break
                    imagens = []
                    for deslocamento in (-0.45, 0.0, 0.45):
                        pos = max(0, min(total - 1, indice + int(round(deslocamento * fps))))
                        captura.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ok, frame = captura.read()
                        if ok and frame is not None:
                            imagens.append(frame)
                    if len(imagens) < 3 or any((i.shape != imagens[0].shape for i in imagens)):
                        continue
                    composta = np.median(np.stack(imagens, axis=0), axis=0).astype(np.uint8)
                    composicoes += 1
                    imagens_compostas[indice] = composta
                    processar(indice, 'COMPOSTO', composta)
        finally:
            captura.release()
            self._estado = None
        arquivos_visuais = {
            'frames': [],
            'avisos': list(avisos_organizacao),
            'pasta_frames': None,
            'pasta_evidencias': None
        }
        if confirmadas and saida_json is not None:
            try:
                gravacao = self._guardar_evidencias_visuais(
                    video,
                    confirmadas,
                    Path(saida_json).parent,
                    imagens_compostas
                )
                arquivos_visuais['frames'] = gravacao['frames']
                arquivos_visuais['avisos'].extend(gravacao['avisos'])
                arquivos_visuais['pasta_frames'] = gravacao['pasta_frames']
                arquivos_visuais['pasta_evidencias'] = gravacao['pasta_evidencias']
            except Exception as erro:
                arquivos_visuais['avisos'].append(f'Falha ao organizar evidências: {type(erro).__name__}: {erro}')
        marca_especifica = next(
            (x['marca'] for x in confirmadas or [] if x['categoria'] != 'DECLARACAO_IA'),
            None
        )
        marca = marca_especifica or (confirmadas[0]['marca'] if confirmadas else None)
        status = 'CONFIRMADA' if confirmadas else 'ERRO' if not analisados or (len(e['erros_ocr']) >= e['chamadas_ocr'] and e['chamadas_ocr']) else 'NAO_CONFIRMADA'
        resultado = {
            'metodo': 'RapidOCR: micro-ROI nativo, mínimo temporal no rodapé, amostragem por lacunas, mediana temporal',
            'status': status,
            'marca_detectada': marca,
            'fps': fps,
            'total_frames': total,
            'duration': duracao,
            'marcas_correlacionadas': sorted(set((x['marca'] for x in confirmadas or []))),
            'classe_marca': next(
                (x['categoria'] for x in confirmadas or [] if x['marca'] == marca),
                None
            ),
            'tipo_confirmacao': tipo_confirmacao,
            'frames_amostrados': len(analisados),
            'frames_planejados': len(iniciais),
            'frames_adaptativos_examinados': len([i for i in analisados if i not in iniciais]),
            'indices_frames_examinados': analisados,
            'indices_adaptativos_planejados': planejados,
            'composicoes_temporais': composicoes,
            'composicoes_minimo_temporal': composicoes_minimo,
            'fontes_composicoes_minimo': minimos_proveniencia,
            'evidencias_confirmadas': confirmadas or [],
            'frames_evidencia': arquivos_visuais['frames'],
            'pasta_frames_evidencia': arquivos_visuais['pasta_frames'],
            'pasta_imagens_evidencia': arquivos_visuais['pasta_evidencias'],
            'avisos_salvamento': arquivos_visuais['avisos'],
            'status_salvamento': 'NAO_APLICAVEL' if not confirmadas else 'COM_AVISOS' if arquivos_visuais['avisos'] else 'OK' if all((e.get('imagem_evidencia') and e.get('frame_original') for e in confirmadas)) else 'INCOMPLETO',
            'evidencias_candidatas': candidatos,
            'textos_ocr_amostrados': e['textos_ocr_amostrados'],
            'chamadas_ocr': e['chamadas_ocr'],
            'limite_ocr_atingido': e['limite_atingido'],
            'limites_configurados': {
                'chamadas_ocr': self.limite_chamadas_ocr,
                'tempo_segundos': tempo_dinamico
            },
            'orientacoes_checadas': list(self.ORIENTACOES),
            'erros': erros + e['erros_ocr'],
            'tempo_segundos': round(time.perf_counter() - inicio, 4),
            'avisos': [
                'A marca OCR não comprova origem criptográfica e pode ser inserida em vídeo real.',
                'Ausência de marca não indica autenticidade: segue-se o Effort.',
                'OCR não verifica marcas invisíveis ou logos puramente gráficos.',
                'A mediana temporal pode apagar marcas móveis; é um método auxiliar.',
                'O mínimo temporal pode apagar marcas móveis ou escuras, e não confirma sozinho autoria de IA.',
                'O orçamento é cooperativo; uma chamada OCR pode exceder o limite de tempo.'
            ]
        }
        if saida_json is not None:
            destino = Path(saida_json)
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(
                json.dumps(resultado, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        return resultado

    @staticmethod
    def como_resultado_visual(resultado, saida_json):
        if resultado.get('status') != 'CONFIRMADA':
            raise ValueError('Somente confirmação pode encerrar o ramo Effort')
        confirmadas = resultado['evidencias_confirmadas']
        n = len(confirmadas)
        visual = {
            'visual_score': 1.0,
            'visual_score_method': 'VISUAL_MARCA_VISIVEL_OCR',
            'visual_screening_method': 'MARCA_VISIVEL_OCR',
            'classification': 'SUSPEITO',
            'classification_reason': 'MARCA_DAGUA_VISIVEL_CONFIRMADA',
            'visual_confidence': None,
            'classification_confidence': None,
            'visual_consistency': 1.0,
            'aggregation_method': 'regra_marca_visual_repetida',
            'fusion_reason': 'EVIDENCIA_VISUAL_MARCA_DAGUA',
            'visual_statistics': {
                'persistencia_40': 1.0,
                'fracao_acima_40': 1.0,
                'fracao_acima_50': 1.0,
                'fracao_acima_80': 1.0
            },
            'frames_analyzed': n,
            'faces_analyzed': 0,
            'frame_scores': [],
            'selected_frame_scores': [],
            'score_effort': None,
            'classification_effort': None,
            'score_effort_guided_full': None,
            'score_effort_guided_face': None,
            'classification_effort_guided_full': None,
            'classification_effort_guided_face': None,
            'marca_dagua_detectada': resultado['marca_detectada'],
            'marca_dagua_confirmada': True,
            'marca_dagua_evidencias': confirmadas,
            'marca_dagua_frames_evidencia': resultado.get('frames_evidencia', []),
            'marca_dagua_pasta_frames': resultado.get('pasta_frames_evidencia'),
            'marca_dagua_pasta_evidencias': resultado.get('pasta_imagens_evidencia'),
            'marca_dagua_status_salvamento': resultado.get('status_salvamento'),
            'marca_dagua_avisos_salvamento': resultado.get('avisos_salvamento', []),
            'marca_dagua_tipo_confirmacao': resultado.get('tipo_confirmacao'),
            'marcas_correlacionadas': resultado.get('marcas_correlacionadas', []),
            'etapa_visual_executada': 'OCR_MARCA_DAGUA',
            'effort_executado': False,
            'runtime_seconds': {
                'watermark': resultado['tempo_segundos'],
                'effort': 0.0,
                'visual_total': resultado['tempo_segundos']
            }
        }
        destino = Path(saida_json)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(visual, ensure_ascii=False, indent=2), encoding='utf-8')
        return visual


import json
from collections import Counter
import math
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvolucaoSeparavel(nn.Module):

    def __init__(self, entrada, saida, kernel=3, passo=1, borda=1):
        super().__init__()
        self.conv1 = nn.Conv2d(entrada, entrada, kernel, passo, borda, groups=entrada, bias=False)
        self.pointwise = nn.Conv2d(entrada, saida, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class BlocoXception(nn.Module):

    def __init__(
        self,
        entrada,
        saida,
        repeticoes,
        passo=1,
        com_relu_inicial=True,
        cresce_primeiro=True
    ):
        super().__init__()
        if saida != entrada or passo != 1:
            self.skip = nn.Conv2d(entrada, saida, 1, passo, bias=False)
            self.skipbn = nn.BatchNorm2d(saida)
        else:
            self.skip = None
        componentes = []
        canal = entrada
        if cresce_primeiro:
            componentes += [
                nn.ReLU(inplace=False),
                ConvolucaoSeparavel(entrada, saida),
                nn.BatchNorm2d(saida)
            ]
            canal = saida
        for _ in range(repeticoes - 1):
            componentes += [
                nn.ReLU(inplace=False),
                ConvolucaoSeparavel(canal, canal),
                nn.BatchNorm2d(canal)
            ]
        if not cresce_primeiro:
            componentes += [
                nn.ReLU(inplace=False),
                ConvolucaoSeparavel(entrada, saida),
                nn.BatchNorm2d(saida)
            ]
        if not com_relu_inicial:
            componentes = componentes[1:]
        if passo != 1:
            componentes.append(nn.MaxPool2d(3, passo, 1))
        self.rep = nn.Sequential(*componentes)

    def forward(self, x):
        residual = x if self.skip is None else self.skipbn(self.skip(x))
        return self.rep(x) + residual


# Define a rede usada para a análise espacial e de fase.
class XceptionSPSL(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = BlocoXception(64, 128, 2, 2, False)
        self.block2 = BlocoXception(128, 256, 2, 2)
        self.block3 = BlocoXception(256, 728, 2, 2)
        for n in range(4, 12):
            setattr(self, f'block{n}', BlocoXception(728, 728, 3))
        self.block12 = BlocoXception(728, 1024, 2, 2, True, False)
        self.conv3 = ConvolucaoSeparavel(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = ConvolucaoSeparavel(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, 2)
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
        )

    def forward(self, imagem_4c):
        x = self.relu(self.bn1(self.conv1(imagem_4c)))
        x = self.relu(self.bn2(self.conv2(x)))
        for n in range(1, 13):
            x = getattr(self, f'block{n}')(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.last_linear(x)


def preparar_entrada_spsl(imagem_bgr):
    rgb = cv2.cvtColor(
        cv2.resize(imagem_bgr, (256, 256), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB
    )
    entrada = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0)
    entrada = entrada.div(255.0).sub(0.5).div(0.5)
    return entrada


def adicionar_canal_fase(entrada):
    cinza = entrada.mean(dim=1, keepdim=True)
    espectro = torch.fft.fftn(cinza, dim=(-1, -2))
    fase = torch.angle(espectro)
    inversa = torch.fft.ifftn(torch.exp(1j * fase), dim=(-1, -2)).real
    return torch.cat([entrada, inversa], dim=1)


def indices_temporais(total, quantidade):
    if total <= 0:
        return []
    return sorted(set((int(round(i)) for i in np.linspace(0, total - 1, min(total, quantidade)))))


# Aplica o SPSL aos quadros selecionados e registra os scores.
class DetectorForenseSPSL:

    def __init__(
        self,
        pesos='models/visual/spsl_best.pth',
        yunet='models/visual/face_detection_yunet_2023mar.onnx',
        max_frames=8,
        face_min_confidence=0.8,
        margem=0.3,
        dispositivo='cpu',
        cpu_threads=None
    ):
        self.pesos = Path(pesos)
        self.yunet = Path(yunet)
        self.max_frames = max(1, int(max_frames))
        self.margem = float(margem)
        self.face_min_confidence = float(face_min_confidence)
        self.device = torch.device(dispositivo)
        self.modelo = None
        self.detector_face = None
        self.cpu_threads = cpu_threads

    def carregar(self):
        if self.modelo is not None:
            return
        if not self.pesos.is_file():
            raise FileNotFoundError(f'Checkpoint SPSL ausente: {self.pesos}. Coloque os pesos neste caminho; enquanto isso, o Effort continua normalmente.')
        if not self.yunet.is_file():
            raise FileNotFoundError(f'Modelo YuNet ausente: {self.yunet}')
        if self.cpu_threads and self.device.type == 'cpu':
            try:
                torch.set_num_threads(max(1, int(self.cpu_threads)))
            except RuntimeError:
                pass
        modelo = XceptionSPSL()
        try:
            dados = torch.load(str(self.pesos), map_location='cpu', weights_only=True)
        except TypeError:
            dados = torch.load(str(self.pesos), map_location='cpu')
        if not isinstance(dados, dict):
            raise RuntimeError('Formato inválido de checkpoint SPSL (esperado state_dict).')
        state = dados
        for chave in ('state_dict', 'model', 'model_state_dict', 'net'):
            if isinstance(state, dict) and chave in state and isinstance(state[chave], dict):
                state = state[chave]
        state = {
            str(k).removeprefix('module.').removeprefix('backbone.'): v for k,
            v in state.items() if torch.is_tensor(v)
        }
        for k, v in list(state.items()):
            if 'pointwise.weight' in k and v.ndim == 2:
                state[k] = v[:, :, None, None]
        if not any((
            k == 'conv1.weight' and tuple(v.shape) == (32, 4, 3, 3) for k,
            v in state.items()
        )):
            raise RuntimeError('Checkpoint não contém convolução SPSL de 4 canais; não use pesos Xception genéricos.')
        if 'last_linear.weight' not in state:
            raise RuntimeError('Checkpoint SPSL não contém cabeça classificadora treinada.')
        ausentes, inesperadas = modelo.load_state_dict(state, strict=False)
        ausentes_relevantes = [x for x in ausentes if not x.startswith('adjust_channel.')]
        inesperadas_relevantes = [x for x in inesperadas if not x.startswith('adjust_channel.')]
        if ausentes_relevantes or inesperadas_relevantes:
            raise RuntimeError(f'Incompatibilidade de checkpoint SPSL. Ausentes={ausentes_relevantes[:8]}; extras={inesperadas_relevantes[:8]}')
        self.modelo = modelo.to(self.device).eval()
        self.detector_face = cv2.FaceDetectorYN_create(
            str(self.yunet),
            '',
            (320, 320),
            self.face_min_confidence,
            0.3,
            5000
        )

    def _face(self, imagem):
        altura, largura = imagem.shape[:2]
        self.detector_face.setInputSize((largura, altura))
        _, rostos = self.detector_face.detect(imagem)
        if rostos is None or len(rostos) == 0:
            return (None, None)
        rosto = max(rostos, key=lambda f: float(f[2] * f[3]))
        x, y, w, h = [float(v) for v in rosto[:4]]
        dx, dy = (w * self.margem, h * self.margem)
        x1, y1 = (max(0, int(x - dx)), max(0, int(y - dy)))
        x2, y2 = (min(largura, int(x + w + dx)), min(altura, int(y + h + dy)))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return (None, None)
        return (imagem[y1:y2, x1:x2], [x1, y1, x2 - x1, y2 - y1])

    def analisar(self, video, destino):
        inicio = time.perf_counter()
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        resultado = {
            'modelo': 'SPSL_DEEPFAKEBENCH_XCEPTION_4C',
            'status': 'NAO_EXECUTADO',
            'modo': 'DIAGNOSTICO',
            'interrompe_effort': False,
            'checkpoint': str(self.pesos),
            'frames_analisados': 0,
            'frames_sem_rosto': 0,
            'scores_frames': [],
            'score_medio': None,
            'score_top25': None,
            'score_spsl': None,
            'classificacao_exploratoria': None,
            'limiar_exploratorio': 0.5,
            'tempo_segundos': 0.0
        }
        capture = None
        try:
            self.carregar()
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f'Não foi possível abrir vídeo para SPSL: {video}')
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            indices = indices_temporais(total, self.max_frames)
            for indice in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, indice)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                recorte, bbox = self._face(frame)
                if recorte is None:
                    resultado['frames_sem_rosto'] += 1
                    continue
                entrada = preparar_entrada_spsl(recorte).to(self.device)
                with torch.inference_mode():
                    entrada = adicionar_canal_fase(entrada)
                    logits = self.modelo(entrada)
                    score = float(torch.softmax(logits, dim=1)[0, 1].item())
                resultado['scores_frames'].append({
                    'frame_index': int(indice),
                    'timestamp': round(indice / fps, 3) if fps > 0 else None,
                    'score': score,
                    'face_bbox': bbox
                })
            valores = sorted([x['score'] for x in resultado['scores_frames']], reverse=True)
            resultado['frames_analisados'] = len(valores)
            if valores:
                n_top = max(1, math.ceil(len(valores) * 0.25))
                resultado['score_medio'] = float(np.mean(valores))
                resultado['score_top25'] = float(np.mean(valores[:n_top]))
                resultado['score_spsl'] = resultado['score_medio']
                resultado['classificacao_exploratoria'] = 'SUSPEITO' if resultado['score_spsl'] >= 0.5 else 'NAO_SUSPEITO'
                resultado['status'] = 'OK'
            else:
                resultado['status'] = 'SEM_ROSTOS_VALIDOS'
        except Exception as erro:
            resultado['status'] = 'INDISPONIVEL'
            resultado['erro'] = f'{type(erro).__name__}: {erro}'
        finally:
            if capture is not None:
                capture.release()
            resultado['tempo_segundos'] = float(time.perf_counter() - inicio)
            destino.write_text(
                json.dumps(resultado, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        return resultado


LIMIAR_MEDIO_PADRAO = 0.95
LIMIAR_FRAME_PADRAO = 0.9
MIN_FRAMES_VALIDOS = 4
MIN_FRAMES_FORTES = 3
FRACAO_MINIMA_FORTE = 0.75
MIN_LADO_CROP_PIXELS = 72
TEMPO_MINIMO_ENTRE_EVIDENCIAS = 1.0


def _numero_01(valor):
    try:
        numero = float(valor)
    except (TypeError, ValueError, OverflowError):
        return None
    return numero if math.isfinite(numero) and 0 <= numero <= 1 else None


def _numero_finito(valor):
    try:
        numero = float(valor)
    except (TypeError, ValueError, OverflowError):
        return None
    return numero if math.isfinite(numero) else None


# Confere se os scores são suficientes para encerrar a análise visual.
def avaliar_atalho_spsl(
    resultado,
    *,
    modo='CASCATA_EXPERIMENTAL',
    limiar_medio=LIMIAR_MEDIO_PADRAO,
    limiar_frame=LIMIAR_FRAME_PADRAO,
    min_frames=MIN_FRAMES_VALIDOS,
    min_frames_fortes=MIN_FRAMES_FORTES,
    fracao_minima=FRACAO_MINIMA_FORTE,
    lado_minimo=MIN_LADO_CROP_PIXELS,
    separacao_minima=TEMPO_MINIMO_ENTRE_EVIDENCIAS
):
    modo = str(modo or '').strip().upper()
    if modo not in {'DESATIVADO', 'DIAGNOSTICO', 'CASCATA_EXPERIMENTAL'}:
        raise ValueError('SPSL_MODO inválido; use DESATIVADO, DIAGNOSTICO ou CASCATA_EXPERIMENTAL')
    if not 0.5 <= float(limiar_medio) <= 1.0:
        raise ValueError('Limiar médio de SPSL fora da faixa segura [0.5, 1]')
    if not 0.5 <= float(limiar_frame) <= 1.0:
        raise ValueError('Limiar de SPSL por frame fora da faixa segura [0.5, 1]')
    resultado = resultado if isinstance(resultado, dict) else {}
    frames = resultado.get('scores_frames') or []
    candidatos = []
    descartes = Counter()
    for frame in frames:
        if not isinstance(frame, dict):
            descartes['FRAME_INVALIDO'] += 1
            continue
        score = _numero_01(frame.get('score'))
        bbox = frame.get('face_bbox')
        timestamp = _numero_finito(frame.get('timestamp'))
        indice = frame.get('frame_index')
        if score is None:
            descartes['SCORE_INVALIDO'] += 1
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            descartes['BBOX_AUSENTE'] += 1
            continue
        largura, altura = (_numero_finito(bbox[2]), _numero_finito(bbox[3]))
        if largura is None or altura is None or min(largura, altura) < lado_minimo:
            descartes['ROSTO_PEQUENO'] += 1
            continue
        if timestamp is None or timestamp < 0:
            descartes['TIMESTAMP_AUSENTE'] += 1
            continue
        if indice is None:
            descartes['FRAME_SEM_INDICE'] += 1
            continue
        candidatos.append({
            'frame_index': str(indice),
            'timestamp': timestamp,
            'score': score,
            'bbox': list(bbox[:4])
        })
    unicos = {}
    for candidato in candidatos:
        indice = candidato['frame_index']
        if indice not in unicos or candidato['score'] > unicos[indice]['score']:
            unicos[indice] = candidato
    validos = sorted(unicos.values(), key=lambda x: (x['timestamp'], x['frame_index']))
    fortes = [f for f in validos if f['score'] >= limiar_frame]
    media_inferida = sum((f['score'] for f in validos)) / len(validos) if validos else None
    media_modelo = _numero_01(resultado.get('score_spsl'))
    motivos = []
    if resultado.get('status') != 'OK':
        motivos.append('SPSL_SEM_INFERENCIA_VALIDA')
    if media_modelo is None:
        motivos.append('SCORE_MEDIO_AUSENTE')
    elif media_modelo < limiar_medio:
        motivos.append('SCORE_MEDIO_ABAIXO_DO_LIMIAR')
    if len(validos) < min_frames:
        motivos.append('FRAMES_COM_ROSTO_QUALIFICADO_INSUFICIENTES')
    if len(fortes) < min_frames_fortes:
        motivos.append('POUCOS_FRAMES_COM_SCORE_ALTO')
    fracao = len(fortes) / len(validos) if validos else 0.0
    if fracao < fracao_minima:
        motivos.append('PERSISTENCIA_INSUFICIENTE')
    duracao = max(
        (f['timestamp'] for f in fortes),
        default=0.0
    ) - min((f['timestamp'] for f in fortes), default=0.0)
    if len(fortes) < 2 or duracao < separacao_minima:
        motivos.append('EVIDENCIAS_SEM_SEPARACAO_TEMPORAL')
    scores_todos = [_numero_01(x.get('score')) for x in frames if isinstance(x, dict)]
    scores_todos = [s for s in scores_todos if s is not None]
    media_todos = sum(scores_todos) / len(scores_todos) if scores_todos else None
    if media_modelo is not None and media_todos is not None and (abs(media_modelo - media_todos) > 0.002):
        motivos.append('SCORE_MEDIO_INCONSISTENTE_COM_FRAMES')
    if modo != 'CASCATA_EXPERIMENTAL':
        motivos.append('MODO_SEM_ATALHO')
    elegivel = len(motivos) == 0
    return {
        'modo': modo,
        'elegivel': bool(elegivel),
        'interrompe_effort': bool(elegivel),
        'limiar_medio': float(limiar_medio),
        'limiar_frame': float(limiar_frame),
        'min_frames': int(min_frames),
        'min_frames_fortes': int(min_frames_fortes),
        'fracao_minima': float(fracao_minima),
        'lado_minimo_crop_pixels': int(lado_minimo),
        'separacao_minima_segundos': float(separacao_minima),
        'score_medio_spsl': media_modelo,
        'media_somente_rostos_qualificados': media_inferida,
        'frames_com_rosto_qualificado': len(validos),
        'frames_fortes': len(fortes),
        'fracao_frames_fortes': fracao,
        'intervalo_entre_evidencias_segundos': float(duracao),
        'indices_evidencias': [f['frame_index'] for f in fortes],
        'frames_descartados': dict(descartes),
        'motivos_nao_atalho': motivos
    }


def como_resultado_visual_spsl(resultado, criterio):
    if not criterio.get('elegivel') or resultado.get('status') != 'OK':
        raise ValueError('SPSL não confirmou evidência suficiente para atalho')
    score = _numero_01(resultado.get('score_spsl'))
    if score is None:
        raise ValueError('Score SPSL ausente')
    frames = []
    for item in resultado.get('scores_frames') or []:
        if not isinstance(item, dict) or _numero_01(item.get('score')) is None:
            continue
        frames.append({
            'frame_index': item.get('frame_index'),
            'timestamp': item.get('timestamp'),
            'visual_score': float(item['score']),
            'face_bbox': item.get('face_bbox')
        })
    if not frames:
        raise ValueError('SPSL não forneceu frames para relatório')
    scores = [f['visual_score'] for f in frames]
    n = len(scores)
    tempo = float(resultado.get('tempo_segundos') or 0)
    return {
        'visual_score': float(score),
        'visual_score_method': 'VISUAL_SPSL_CASCATA_EXPERIMENTAL',
        'visual_screening_method': 'MARCA_SPSL_EFFORT',
        'classification': 'SUSPEITO',
        'classification_reason': 'SPSL_EVIDENCIA_ALTA_E_PERSISTENTE',
        'visual_confidence': None,
        'classification_confidence': None,
        'visual_consistency': float(criterio['fracao_frames_fortes']),
        'aggregation_method': 'spsl_media_com_consistencia_temporal',
        'fusion_reason': 'EVIDENCIA_FORENSE_SPSL_CONFIRMACAO_TEMPORAL',
        'visual_statistics': {
            'persistencia_40': sum((s >= 0.4 for s in scores)) / n,
            'fracao_acima_40': sum((s >= 0.4 for s in scores)) / n,
            'fracao_acima_50': sum((s >= 0.5 for s in scores)) / n,
            'fracao_acima_80': sum((s >= 0.8 for s in scores)) / n
        },
        'frames_analyzed': n,
        'faces_analyzed': n,
        'frame_scores': frames,
        'selected_frame_scores': frames,
        'score_effort': None,
        'classification_effort': None,
        'score_effort_guided_full': None,
        'score_effort_guided_face': None,
        'classification_effort_guided_full': None,
        'classification_effort_guided_face': None,
        'marca_dagua_confirmada': False,
        'marca_dagua_detectada': None,
        'etapa_visual_executada': 'SPSL_CASCATA_EXPERIMENTAL',
        'effort_executado': False,
        'spsl_executado': True,
        'spsl_interrompeu_effort': True,
        'spsl_diagnostico': resultado,
        'criterios_atalho_spsl': criterio,
        'runtime_seconds': {'spsl': tempo, 'effort': 0.0, 'visual_total': tempo}
    }


import gc
import json
import math
import os
import tempfile
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# Define a parte de classificação usada pelo modelo Effort.
class EffortCompactDetector(nn.Module):

    def __init__(
        self,
        clip_model_name='openai/clip-vit-large-patch14',
        cache_dir='models/visual/huggingface'
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
        except ImportError as error:
            raise RuntimeError("A dependencia 'transformers' e necessaria para o Effort. Instale com: python -m pip install -U transformers") from error
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f'Carregando backbone CLIP ViT-L/14 para Effort: {clip_model_name}')
        try:
            from transformers.utils import logging as hf_logging
            nivel_hf_anterior = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            hf_logging = None
            nivel_hf_anterior = None
        try:
            self.backbone = CLIPVisionModel.from_pretrained(
                clip_model_name,
                cache_dir=str(cache_dir)
            )
        finally:
            if hf_logging is not None and nivel_hf_anterior is not None:
                hf_logging.set_verbosity(nivel_hf_anterior)
        self.head = nn.Linear(1024, 2)

    def forward(self, x):
        saida = self.backbone(pixel_values=x, return_dict=True)
        return self.head(saida.pooler_output)


# Organiza a análise do Effort em quadros e regiões faciais.
class VisualDeepfakeDetector:
    SCORE_METHOD = 'VISUAL_EFFORT_FACE_THRESHOLD_ROUTED'
    SCREENING_METHOD = 'VISUAL_EFFORT_SCENE_FACE_GUIDED_DUALVIEW'
    AGGREGATION_METHOD = 'effort_top25_face_threshold_scene_face_guided_dualview'
    YUNET_DOWNLOAD_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

    def __init__(
        self,
        effort_weights_path='models/visual/effort_clip_L14_trainOn_sdv14.pth',
        effort_clip_model_name='openai/clip-vit-large-patch14',
        hf_cache_dir='models/visual/huggingface',
        output_path='output/visual_score.json',
        limite_nao_suspeito=0.35,
        limite_suspeito=0.65,
        consistencia_minima_suspeito=0.65,
        max_frames=8,
        yunet_model_path='models/visual/face_detection_yunet_2023mar.onnx',
        yunet_score_threshold=0.8,
        yunet_nms_threshold=0.3,
        yunet_input_max_side=640,
        guided_scan_max_frames=60,
        guided_max_frames=12,
        guided_min_face_side=40,
        guided_face_margin=0.3,
        scene_bhattacharyya_min=0.55,
        scene_mad_min=0.12
    ):
        self.effort_weights_path = Path(effort_weights_path)
        self.effort_clip_model_name = str(effort_clip_model_name)
        self.hf_cache_dir = Path(hf_cache_dir)
        self.output_path = Path(output_path)
        self.limite_nao_suspeito = float(limite_nao_suspeito)
        self.limite_suspeito = float(limite_suspeito)
        self.consistencia_minima_suspeito = float(consistencia_minima_suspeito)
        self.max_frames = max(1, int(max_frames))
        self.yunet_model_path = Path(yunet_model_path)
        self.yunet_score_threshold = float(yunet_score_threshold)
        self.yunet_nms_threshold = float(yunet_nms_threshold)
        self.yunet_input_max_side = max(160, int(yunet_input_max_side))
        self.guided_scan_max_frames = max(8, int(guided_scan_max_frames))
        self.guided_max_frames = max(1, int(guided_max_frames))
        self.guided_min_face_side = max(16, int(guided_min_face_side))
        self.guided_face_margin = max(0.0, float(guided_face_margin))
        self.scene_bhattacharyya_min = float(scene_bhattacharyya_min)
        self.scene_mad_min = float(scene_mad_min)
        self._validar_configuracao()
        self._validar_arquivos()
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = self.device.type == 'cuda'
        print(f'Dispositivo visual: {self.device}')
        self.effort_model = EffortCompactDetector(
            clip_model_name=self.effort_clip_model_name,
            cache_dir=self.hf_cache_dir
        )
        self._load_effort_weights_compact()
        self.effort_model.to(self.device).eval()
        if self.use_fp16:
            self.effort_model.half()
        self._effort_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            dtype=torch.float32
        ).view(3, 1, 1)
        self._effort_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            dtype=torch.float32
        ).view(3, 1, 1)
        self._garantir_yunet()
        self._yunet = self._carregar_yunet()

    def _validar_configuracao(self):
        if not 0.0 <= self.limite_nao_suspeito < self.limite_suspeito <= 1.0:
            raise ValueError('Limites visuais Effort invalidos.')
        if not 0.0 <= self.consistencia_minima_suspeito <= 1.0:
            raise ValueError('Consistencia minima Effort invalida.')
        if not 0.0 <= self.yunet_score_threshold <= 1.0:
            raise ValueError('Threshold YuNet invalido.')
        if not 0.0 <= self.scene_bhattacharyya_min <= 1.0:
            raise ValueError('Threshold de cena Bhattacharyya invalido.')
        if not 0.0 <= self.scene_mad_min <= 1.0:
            raise ValueError('Threshold de cena MAD invalido.')

    def _validar_arquivos(self):
        if not self.effort_weights_path.exists():
            raise FileNotFoundError(f'Checkpoint Effort/GenImage nao encontrado: {self.effort_weights_path}')

    @staticmethod
    def _torch_load_compat(caminho, mmap=False):
        caminho = str(caminho)
        tentativas = []
        if mmap:
            tentativas.extend([
                {'map_location': 'cpu', 'weights_only': False, 'mmap': True},
                {'map_location': 'cpu', 'mmap': True}
            ])
        tentativas.extend([{'map_location': 'cpu', 'weights_only': False}, {'map_location': 'cpu'}])
        ultimo_erro = None
        for kwargs in tentativas:
            try:
                return torch.load(caminho, **kwargs)
            except TypeError as error:
                ultimo_erro = error
        if ultimo_erro is not None:
            raise ultimo_erro
        raise RuntimeError(f'Nao foi possivel carregar checkpoint: {caminho}')

    @staticmethod
    def _normalizar_state_dict(state_dict):
        corrigido = {}
        for chave, valor in state_dict.items():
            nova_chave = str(chave)
            for prefixo in ('module.', 'model.'):
                if nova_chave.startswith(prefixo):
                    nova_chave = nova_chave[len(prefixo):]
            corrigido[nova_chave] = valor
        return corrigido

    @staticmethod
    def _extrair_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint
        for chave in ('state_dict', 'model', 'model_state_dict'):
            candidato = checkpoint.get(chave)
            if isinstance(candidato, dict):
                return candidato
        return checkpoint

    @staticmethod
    def _reconstruir_peso_effort(state_dict, prefixo):
        weight_main = state_dict.get(prefixo + '.weight_main')
        if weight_main is None:
            return None
        peso = weight_main.float()
        u = state_dict.get(prefixo + '.U_residual')
        s = state_dict.get(prefixo + '.S_residual')
        v = state_dict.get(prefixo + '.V_residual')
        if u is not None and s is not None and (v is not None) and (s.numel() > 0):
            residual = u.float() * s.float().view(1, -1) @ v.float()
            peso = peso + residual
        return peso

    @staticmethod
    def _chaves_checkpoint_effort(chave_destino):
        candidatos = [chave_destino]
        prefixo_novo = 'backbone.vision_model.'
        prefixo_antigo = 'backbone.'
        if chave_destino.startswith(prefixo_novo):
            candidatos.append(prefixo_antigo + chave_destino[len(prefixo_novo):])
        elif chave_destino.startswith(prefixo_antigo):
            candidatos.append(prefixo_novo + chave_destino[len(prefixo_antigo):])
        return candidatos

    def _load_effort_weights_compact(self):
        print(f'Carregando checkpoint Effort/GenImage: {self.effort_weights_path}')
        checkpoint = self._torch_load_compat(self.effort_weights_path, mmap=True)
        state_dict = self._normalizar_state_dict(self._extrair_state_dict(checkpoint))
        if not isinstance(state_dict, dict):
            raise RuntimeError('Checkpoint Effort inválido: não contém um dicionário de pesos.')
        destino = self.effort_model.state_dict()
        compactado = {}
        reconstruidos = 0
        diretos = 0
        chaves_remapeadas = 0
        ausentes = []
        incompatíveis = []
        sufixos_projecao = (
            '.self_attn.q_proj.weight',
            '.self_attn.k_proj.weight',
            '.self_attn.v_proj.weight',
            '.self_attn.out_proj.weight',
            '.mlp.fc1.weight',
            '.mlp.fc2.weight'
        )
        projecoes_esperadas = [chave for chave in destino if chave.endswith(sufixos_projecao)]
        if not projecoes_esperadas:
            raise RuntimeError('Estrutura do CLIP visual inesperada: nenhuma projeção encontrada.')
        for chave_destino, tensor_destino in destino.items():
            tensor = None
            reconstruido = False
            origem = None
            for chave_origem in self._chaves_checkpoint_effort(chave_destino):
                if chave_origem in state_dict:
                    tensor = state_dict[chave_origem]
                    origem = chave_origem
                    break
                if chave_origem.endswith('.weight'):
                    prefixo = chave_origem[:-len('.weight')]
                    if prefixo + '.weight_main' in state_dict:
                        tensor = self._reconstruir_peso_effort(state_dict, prefixo)
                        reconstruido = tensor is not None
                        origem = chave_origem
                        break
            if tensor is None:
                ausentes.append(chave_destino)
                continue
            if not hasattr(tensor, 'shape') or tensor.shape != tensor_destino.shape:
                incompatíveis.append((chave_destino, origem, tuple(tensor.shape)))
                continue
            compactado[chave_destino] = tensor.float() if tensor.is_floating_point() else tensor
            if origem != chave_destino:
                chaves_remapeadas += 1
            if reconstruido:
                reconstruidos += 1
            else:
                diretos += 1
        if 'head.weight' not in compactado or 'head.bias' not in compactado:
            raise RuntimeError(f'Checkpoint Effort incompatível: cabeça de classificação ausente ou com dimensão diferente ({self.effort_weights_path}).')
        projecoes_carregadas = sum((chave in compactado for chave in projecoes_esperadas))
        minimo_projecoes = min(80, len(projecoes_esperadas))
        if projecoes_carregadas < minimo_projecoes:
            amostra_projecoes = [chave for chave in state_dict if '.self_attn.' in chave or '.mlp.' in chave][:5]
            raise RuntimeError(f'Checkpoint Effort incompatível ou incompleto: {projecoes_carregadas}/{len(projecoes_esperadas)} projeções carregadas (mínimo {minimo_projecoes}). Arquivo: {self.effort_weights_path}. Chaves de camadas presentes: {amostra_projecoes or list(state_dict)[:5]}.')
        if incompatíveis:
            raise RuntimeError(f'Checkpoint Effort com dimensões divergentes em {len(incompatíveis)} tensores: {incompatíveis[:3]}.')
        missing, unexpected = self.effort_model.load_state_dict(compactado, strict=False)
        if any((chave.startswith('head.') for chave in missing)):
            raise RuntimeError('Falha ao carregar a cabeça de classificação do Effort.')
        print(f'Effort carregado: {reconstruidos} pesos SVD reconstruídos, {diretos} tensores diretos, {chaves_remapeadas} chaves com prefixos ajustados, {projecoes_carregadas}/{len(projecoes_esperadas)} projeções, {len(ausentes)} tensores mantidos do CLIP original.')
        if unexpected:
            print(f'Aviso Effort: {len(unexpected)} chaves inesperadas.')
        del checkpoint, state_dict, compactado
        gc.collect()

    def _garantir_yunet(self):
        if self.yunet_model_path.exists() and self.yunet_model_path.stat().st_size > 100000:
            return
        self.yunet_model_path.parent.mkdir(parents=True, exist_ok=True)
        print('Modelo YuNet nao encontrado. Baixando modelo oficial do OpenCV Zoo...')
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.onnx',
                delete=False,
                dir=str(self.yunet_model_path.parent)
            ) as tmp:
                tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(self.YUNET_DOWNLOAD_URL, str(tmp_path))
            if not tmp_path.exists() or tmp_path.stat().st_size <= 100000:
                raise RuntimeError('Arquivo YuNet baixado parece invalido ou incompleto.')
            os.replace(str(tmp_path), str(self.yunet_model_path))
            print(f'YuNet salvo em: {self.yunet_model_path}')
        except Exception as error:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise RuntimeError(f"Nao foi possivel baixar o YuNet automaticamente. Baixe manualmente 'face_detection_yunet_2023mar.onnx' do OpenCV Zoo e salve em models/visual/. Erro original: {error}") from error

    def _carregar_yunet(self):
        if not hasattr(cv2, 'FaceDetectorYN'):
            raise RuntimeError('O OpenCV instalado nao possui cv2.FaceDetectorYN. Atualize com: python -m pip install -U opencv-python')
        try:
            detector = cv2.FaceDetectorYN.create(
                str(self.yunet_model_path),
                '',
                (320, 320),
                self.yunet_score_threshold,
                self.yunet_nms_threshold,
                5000
            )
        except Exception as error:
            raise RuntimeError(f'Falha ao carregar YuNet: {self.yunet_model_path}. Erro: {error}') from error
        print(f'YuNet pronto para selecao facial | threshold={self.yunet_score_threshold:.2f}')
        return detector

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))

    @staticmethod
    def _top_fraction_mean_values(values, fraction):
        if not values:
            return None
        quantidade = max(1, math.ceil(len(values) * float(fraction)))
        ordenados = sorted([float(v) for v in values], reverse=True)
        return float(np.mean(ordenados[:quantidade]))

    @staticmethod
    def _selecionar_uniforme(lista, quantidade):
        itens = list(lista or [])
        if not itens:
            return []
        quantidade = min(len(itens), max(1, int(quantidade)))
        if quantidade == len(itens):
            return itens
        indices = np.linspace(0, len(itens) - 1, num=quantidade)
        indices = sorted(set((int(round(v)) for v in indices)))
        return [itens[i] for i in indices]

    def _selecionar_frames_baseline(self, frame_data):
        return self._selecionar_uniforme(
            list((frame_data or {}).get('frames', []) or []),
            self.max_frames
        )

    @staticmethod
    def _abrir_frame_rgb(frame_info):
        caminho = Path(frame_info['path'])
        with Image.open(caminho) as image:
            return image.convert('RGB')

    @staticmethod
    def _pil_para_tensor_01(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(array)).float()

    def _preparar_effort_image(self, image):
        original_width, original_height = image.size
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        processada = image.resize((224, 224), resample=resample)
        tensor = self._pil_para_tensor_01(processada)
        tensor = (tensor - self._effort_mean) / self._effort_std
        return (
            tensor.unsqueeze(0),
            {
                'original_width': int(original_width),
                'original_height': int(original_height),
                'processed_width': 224,
                'processed_height': 224,
                'resize_method': 'PIL_BILINEAR_224X224',
                'normalization': 'CLIP_MEAN_STD'
            }
        )

    def _predict_effort_image(self, image):
        tensor, meta = self._preparar_effort_image(image)
        tensor = tensor.to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        with torch.inference_mode():
            logits = self.effort_model(tensor)
            score = torch.softmax(logits.float(), dim=1)[0, 1].item()
        return (float(score), meta)

    def _predict_effort_full(self, frame_info):
        image = self._abrir_frame_rgb(frame_info)
        return self._predict_effort_image(image)

    def _calcular_estatisticas(self, values):
        values = [float(v) for v in values if v is not None]
        if not values:
            return None
        return {
            'score_video': float(np.mean(values)),
            'media': float(np.mean(values)),
            'mediana': float(np.median(values)),
            'minimo': float(np.min(values)),
            'maximo': float(np.max(values)),
            'p75': self._percentile(values, 75),
            'p90': self._percentile(values, 90),
            'p95': self._percentile(values, 95),
            'top25_mean': self._top_fraction_mean_values(values, 0.25),
            'desvio_padrao': float(np.std(values)),
            'fracao_acima_50': float(sum((v >= 0.5 for v in values)) / len(values)),
            'fracao_acima_65': float(sum((v >= 0.65 for v in values)) / len(values)),
            'quantidade': int(len(values))
        }

    def _consistencia_values(self, values):
        values = [float(v) for v in values if v is not None]
        if not values:
            return 0.0
        return self._clamp(1.0 - float(np.std(values)) / 0.5)

    def _usa_checkpoint_facial(self):
        checkpoint_name = self.effort_weights_path.stem.lower()
        return 'trainon_ff' in checkpoint_name or 'trainon_faceforensic' in checkpoint_name

    def _limiar_binario_effort(self):
        if self._usa_checkpoint_facial():
            return self.limite_suspeito
        return (self.limite_nao_suspeito + self.limite_suspeito) / 2.0

    def _classificar_effort(self, score, consistencia):
        if score is None:
            return ('SUSPEITO', 'SEM_SCORE_EFFORT_VALIDO_TRATADO_COMO_SUSPEITO')
        limiar_binario = self._limiar_binario_effort()
        if score >= limiar_binario:
            if consistencia >= self.consistencia_minima_suspeito:
                return ('SUSPEITO', 'SCORE_ACIMA_DO_LIMIAR_BINARIO')
            return ('SUSPEITO', 'SCORE_ACIMA_DO_LIMIAR_COM_BAIXA_CONSISTENCIA')
        return ('NAO_SUSPEITO', 'SCORE_ABAIXO_DO_LIMIAR_BINARIO')

    def _confianca_effort(self, score, classification, n, max_n):
        if score is None:
            return 0.0
        cobertura = self._clamp(n / max(max_n, 1))
        limiar_binario = self._limiar_binario_effort()
        if classification == 'SUSPEITO':
            base = self._clamp((score - limiar_binario) / max(1.0 - limiar_binario, 1e-06))
        else:
            base = self._clamp((limiar_binario - score) / max(limiar_binario, 1e-06))
        return self._clamp(0.25 * cobertura + 0.75 * base)

    @staticmethod
    def _resize_detection(image_bgr, max_side):
        h, w = image_bgr.shape[:2]
        maior = max(h, w)
        if maior <= max_side:
            return (image_bgr, 1.0)
        escala = float(max_side) / float(maior)
        novo_w = max(1, int(round(w * escala)))
        novo_h = max(1, int(round(h * escala)))
        return (cv2.resize(image_bgr, (novo_w, novo_h)), escala)

    @staticmethod
    def _histograma_hsv(image_bgr):
        small = cv2.resize(image_bgr, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (hist, gray)

    def _mudanca_cena(self, prev_hist, prev_gray, hist, gray):
        if prev_hist is None or prev_gray is None:
            return (False, None)
        distancia = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        mad = float(np.mean(np.abs(prev_gray - gray)) / 255.0)
        corte = distancia >= self.scene_bhattacharyya_min and mad >= self.scene_mad_min
        return (corte, {'bhattacharyya': distancia, 'mean_abs_difference': mad})

    def _detectar_maior_face(self, image_bgr):
        det_img, escala = self._resize_detection(image_bgr, self.yunet_input_max_side)
        h, w = det_img.shape[:2]
        try:
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(det_img)
        except Exception:
            self._yunet = cv2.FaceDetectorYN.create(
                str(self.yunet_model_path),
                '',
                (w, h),
                self.yunet_score_threshold,
                self.yunet_nms_threshold,
                5000
            )
            _, faces = self._yunet.detect(det_img)
        if faces is None or len(faces) == 0:
            return None
        melhor = None
        melhor_area = -1.0
        for row in faces:
            x, y, fw, fh = [float(v) for v in row[:4]]
            if fw <= 0 or fh <= 0:
                continue
            area = fw * fh
            if area > melhor_area:
                score = float(row[-1]) if len(row) >= 15 else None
                melhor = (x, y, fw, fh, score)
                melhor_area = area
        if melhor is None:
            return None
        x, y, fw, fh, score = melhor
        inv = 1.0 / max(escala, 1e-09)
        x *= inv
        y *= inv
        fw *= inv
        fh *= inv
        H, W = image_bgr.shape[:2]
        x = max(0.0, min(float(W - 1), x))
        y = max(0.0, min(float(H - 1), y))
        fw = max(1.0, min(float(W) - x, fw))
        fh = max(1.0, min(float(H) - y, fh))
        return {'bbox': [float(x), float(y), float(fw), float(fh)], 'detector_score': score}

    @staticmethod
    def _metricas_face(image_bgr, bbox):
        x, y, w, h = bbox
        H, W = image_bgr.shape[:2]
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(W, int(math.ceil(x + w)))
        y2 = min(H, int(math.ceil(y + h)))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area_ratio = float(max(w, 0.0) * max(h, 0.0) / max(W * H, 1))
        cx = x + w / 2.0
        cy = y + h / 2.0
        dx = abs(cx - W / 2.0) / max(W / 2.0, 1.0)
        dy = abs(cy - H / 2.0) / max(H / 2.0, 1.0)
        centrality = max(0.0, 1.0 - math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))
        return {
            'face_area_ratio': area_ratio,
            'face_sharpness': sharpness,
            'face_centrality': float(centrality),
            'face_width': float(w),
            'face_height': float(h)
        }

    def _scan_face_guided(self, frame_data):
        frames_all = list((frame_data or {}).get('frames', []) or [])
        frames_scan = self._selecionar_uniforme(frames_all, self.guided_scan_max_frames)
        candidatos = []
        scan_records = []
        scene_id = 0
        prev_hist = None
        prev_gray = None
        faces_detectadas = 0
        print(f'Selecao scene/face-guided: scan={len(frames_scan)} de {len(frames_all)} frames extraidos')
        for idx, frame_info in enumerate(frames_scan):
            path = Path(frame_info['path'])
            image_bgr = cv2.imread(str(path))
            if image_bgr is None:
                scan_records.append({
                    'frame_index': frame_info.get('frame_index'),
                    'timestamp': frame_info.get('timestamp'),
                    'path': str(path),
                    'scene_id': int(scene_id),
                    'face_found': False,
                    'error': 'FRAME_NAO_LIDO'
                })
                continue
            hist, gray = self._histograma_hsv(image_bgr)
            corte, cut_meta = self._mudanca_cena(prev_hist, prev_gray, hist, gray)
            if idx > 0 and corte:
                scene_id += 1
            prev_hist = hist
            prev_gray = gray
            face = self._detectar_maior_face(image_bgr)
            record = {
                'frame_index': int(frame_info['frame_index']),
                'timestamp': float(frame_info['timestamp']),
                'path': str(path),
                'scene_id': int(scene_id),
                'scene_cut_before': bool(corte),
                'scene_cut_diagnostic': cut_meta,
                'face_found': bool(face is not None)
            }
            if face is not None:
                metrics = self._metricas_face(image_bgr, face['bbox'])
                if metrics is not None:
                    record.update(face)
                    record.update(metrics)
                    faces_detectadas += 1
                    if metrics['face_width'] >= self.guided_min_face_side and metrics['face_height'] >= self.guided_min_face_side:
                        candidatos.append(record.copy())
                    else:
                        record['face_rejected_reason'] = 'FACE_PEQUENA'
            scan_records.append(record)
        scene_count = scene_id + 1 if frames_scan else 0
        return {
            'scan_records': scan_records,
            'candidates': candidatos,
            'scan_frames': len(frames_scan),
            'source_frames': len(frames_all),
            'faces_detected': int(faces_detectadas),
            'face_detection_rate': float(faces_detectadas / len(frames_scan)) if frames_scan else 0.0,
            'scene_count': int(scene_count)
        }

    @staticmethod
    def _candidate_quality_key(item):
        return (
            float(item.get('face_area_ratio') or 0.0),
            float(item.get('face_sharpness') or 0.0),
            float(item.get('detector_score') or 0.0),
            float(item.get('face_centrality') or 0.0)
        )

    def _selecionar_guided_candidates(self, scan_result, frame_data):
        candidatos = list(scan_result.get('candidates', []) or [])
        max_n = self.guided_max_frames
        if not candidatos:
            fallback = self._selecionar_uniforme(
                list((frame_data or {}).get('frames', []) or []),
                max_n
            )
            return ([], fallback, 'SEM_FACE_VALIDADA_FALLBACK_FULL_UNIFORME')
        por_scene = {}
        for item in candidatos:
            por_scene.setdefault(int(item['scene_id']), []).append(item)
        scene_bests = []
        for sid in sorted(por_scene):
            melhor = max(por_scene[sid], key=self._candidate_quality_key)
            melhor = dict(melhor)
            melhor['selection_reason'] = 'MELHOR_FACE_DA_CENA'
            scene_bests.append(melhor)
        selecionados = []
        usados = set()
        if len(scene_bests) > max_n:
            indices = np.linspace(0, len(scene_bests) - 1, num=max_n)
            indices = sorted(set((int(round(x)) for x in indices)))
            for i in indices:
                item = dict(scene_bests[i])
                item['selection_reason'] = 'CENAS_DISTRIBUIDAS_NO_TEMPO'
                selecionados.append(item)
                usados.add(str(item['path']))
        else:
            for item in scene_bests:
                selecionados.append(dict(item))
                usados.add(str(item['path']))
        restantes = [dict(x) for x in candidatos if str(x['path']) not in usados]
        while len(selecionados) < max_n and restantes:
            if selecionados:
                tempos = [float(x['timestamp']) for x in selecionados]

                def fill_key(item):
                    t = float(item['timestamp'])
                    distancia = min((abs(t - s) for s in tempos))
                    return (float(distancia), *self._candidate_quality_key(item))
            else:

                def fill_key(item):
                    return (0.0, *self._candidate_quality_key(item))
            melhor = max(restantes, key=fill_key)
            melhor['selection_reason'] = 'COBERTURA_TEMPORAL_COMPLEMENTAR'
            selecionados.append(melhor)
            usados.add(str(melhor['path']))
            restantes = [x for x in restantes if str(x['path']) not in usados]
        selecionados.sort(key=lambda x: float(x['timestamp']))
        guided_full_frames = [{
            'frame_index': int(x['frame_index']),
            'timestamp': float(x['timestamp']),
            'path': Path(x['path'])
        } for x in selecionados]
        return (selecionados, guided_full_frames, 'FACE_GUIDED')

    def _crop_face_with_margin(self, image, bbox):
        W, H = image.size
        x, y, w, h = [float(v) for v in bbox]
        cx = x + w / 2.0
        cy = y + h / 2.0
        side = max(w, h) * (1.0 + 2.0 * self.guided_face_margin)
        side = max(side, float(self.guided_min_face_side))
        x1 = int(math.floor(cx - side / 2.0))
        y1 = int(math.floor(cy - side / 2.0))
        x2 = int(math.ceil(cx + side / 2.0))
        y2 = int(math.ceil(cy + side / 2.0))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            raise RuntimeError('CROP_FACE_INVALIDO')
        crop = image.crop((x1, y1, x2, y2))
        return (crop, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    def _executar_baseline_uniforme(self, frames_selecionados, full_cache):
        saida = []
        erros = []
        inicio = time.perf_counter()
        for index, frame_info in enumerate(frames_selecionados, start=1):
            chave = str(Path(frame_info['path']).resolve())
            try:
                if chave in full_cache:
                    score, meta = full_cache[chave]
                else:
                    score, meta = self._predict_effort_full(frame_info)
                    full_cache[chave] = (score, meta)
                item = {
                    'frame_index': int(frame_info['frame_index']),
                    'timestamp': float(frame_info['timestamp']),
                    'frame_path': str(frame_info['path']),
                    'effort_score': float(score),
                    'visual_score': float(score),
                    **meta
                }
                saida.append(item)
                print(f"Baseline {index:02d}/{len(frames_selecionados):02d} | {item['timestamp']:.2f}s | score={score:.4f}")
            except Exception as error:
                erros.append({
                    'frame_index': frame_info.get('frame_index'),
                    'timestamp': frame_info.get('timestamp'),
                    'erro': str(error)
                })
                print(f'Erro Effort baseline no frame {index}: {error}')
        return (saida, erros, float(time.perf_counter() - inicio))

    def _executar_guided_full(self, frames, full_cache):
        saida = []
        erros = []
        inicio = time.perf_counter()
        for index, frame_info in enumerate(frames, start=1):
            chave = str(Path(frame_info['path']).resolve())
            try:
                if chave in full_cache:
                    score, meta = full_cache[chave]
                    cache_hit = True
                else:
                    score, meta = self._predict_effort_full(frame_info)
                    full_cache[chave] = (score, meta)
                    cache_hit = False
                saida.append({
                    'frame_index': int(frame_info['frame_index']),
                    'timestamp': float(frame_info['timestamp']),
                    'frame_path': str(frame_info['path']),
                    'effort_score': float(score),
                    'cache_hit_baseline': bool(cache_hit),
                    **meta
                })
                print(f"Guided-full {index:02d}/{len(frames):02d} | {float(frame_info['timestamp']):.2f}s | score={score:.4f}")
            except Exception as error:
                erros.append({
                    'frame_index': frame_info.get('frame_index'),
                    'timestamp': frame_info.get('timestamp'),
                    'erro': str(error)
                })
                print(f'Erro Effort guided-full no frame {index}: {error}')
        return (saida, erros, float(time.perf_counter() - inicio))

    def _executar_guided_face(self, selected_candidates):
        saida = []
        erros = []
        inicio = time.perf_counter()
        for index, item in enumerate(selected_candidates, start=1):
            try:
                image = self._abrir_frame_rgb(item)
                crop, crop_bbox = self._crop_face_with_margin(image, item['bbox'])
                score, meta = self._predict_effort_image(crop)
                registro = {
                    'frame_index': int(item['frame_index']),
                    'timestamp': float(item['timestamp']),
                    'frame_path': str(item['path']),
                    'scene_id': int(item['scene_id']),
                    'selection_reason': item.get('selection_reason'),
                    'face_bbox_original': [float(v) for v in item['bbox']],
                    'face_crop_bbox': crop_bbox,
                    'face_area_ratio': float(item.get('face_area_ratio') or 0.0),
                    'face_sharpness': float(item.get('face_sharpness') or 0.0),
                    'face_centrality': float(item.get('face_centrality') or 0.0),
                    'face_detector_score': float(item['detector_score']) if item.get('detector_score') is not None else None,
                    'effort_face_score': float(score),
                    **meta
                }
                saida.append(registro)
                print(f"Guided-face {index:02d}/{len(selected_candidates):02d} | {registro['timestamp']:.2f}s | score={score:.4f} | face={registro['face_area_ratio']:.3f}")
            except Exception as error:
                erros.append({
                    'frame_index': item.get('frame_index'),
                    'timestamp': item.get('timestamp'),
                    'erro': str(error)
                })
                print(f'Erro Effort guided-face no frame {index}: {error}')
        return (saida, erros, float(time.perf_counter() - inicio))

    def _resumo_branch(self, values, max_n):
        stats = self._calcular_estatisticas(values)
        if stats is None:
            return {
                'score': None,
                'classification': 'SUSPEITO',
                'classification_reason': 'SEM_SCORE_VALIDO_TRATADO_COMO_SUSPEITO',
                'consistency': 0.0,
                'confidence': 0.0,
                'statistics': {}
            }
        score = float(stats['top25_mean'])
        consistency = self._consistencia_values(values)
        classification, reason = self._classificar_effort(score, consistency)
        confidence = self._confianca_effort(score, classification, len(values), max_n)
        return {
            'score': score,
            'classification': classification,
            'classification_reason': reason,
            'consistency': float(consistency),
            'confidence': float(confidence),
            'statistics': stats
        }

    def _resultado_sem_score(self, reason):
        return {
            'visual_score': None,
            'visual_score_method': self.SCORE_METHOD,
            'visual_screening_method': self.SCREENING_METHOD,
            'classification': 'SUSPEITO',
            'classification_reason': reason,
            'visual_confidence': 0.0,
            'classification_confidence': 0.0,
            'visual_consistency': 0.0,
            'aggregation_method': self.AGGREGATION_METHOD,
            'fusion_reason': 'EFFORT_DUALVIEW_SEM_FUSAO',
            'score_effort': None,
            'score_effort_mean': None,
            'classification_effort': 'SUSPEITO',
            'score_effort_guided_full': None,
            'classification_effort_guided_full': 'SUSPEITO',
            'score_effort_guided_face': None,
            'classification_effort_guided_face': 'SUSPEITO',
            'frame_scores_effort': [],
            'frame_scores_effort_guided_full': [],
            'frame_scores_effort_guided_face': [],
            'frame_scores': [],
            'selected_frame_scores': [],
            'frames_analyzed': 0,
            'faces_analyzed': 0,
            'runtime_seconds': {
                'effort_baseline': 0.0,
                'face_scene_scan': 0.0,
                'effort_guided_full': 0.0,
                'effort_guided_face': 0.0,
                'effort': 0.0,
                'visual_total': 0.0
            }
        }

    def process(self, face_data=None, frame_data=None, video_path=None):
        inicio_total = time.perf_counter()
        baseline_frames = self._selecionar_frames_baseline(frame_data)
        if not baseline_frames:
            print('Nenhum frame disponivel para analise Effort.')
            return self._resultado_sem_score('SEM_DADOS_VISUAIS')
        print()
        print('Executando modalidade visual: Effort dual-view.')
        print('A decisao primaria depende do dominio do checkpoint Effort.')
        print('Screening: selecao scene/face-guided + frame inteiro + crop facial.')
        print('As tres leituras sao registradas separadamente, sem fusao entre elas.')
        print('-' * 76)
        full_cache = {}
        baseline_scores, baseline_errors, tempo_baseline = self._executar_baseline_uniforme(baseline_frames, full_cache)
        baseline_values = [x['effort_score'] for x in baseline_scores]
        baseline = self._resumo_branch(baseline_values, self.max_frames)
        if baseline['score'] is None:
            result = self._resultado_sem_score('SEM_SCORE_EFFORT_VALIDO')
            result['effort_errors'] = baseline_errors
            result['runtime_seconds']['effort_baseline'] = tempo_baseline
            result['runtime_seconds']['effort'] = tempo_baseline
            result['runtime_seconds']['visual_total'] = float(time.perf_counter() - inicio_total)
            return result
        inicio_scan = time.perf_counter()
        scan = self._scan_face_guided(frame_data)
        selected_candidates, guided_full_frames, selection_mode = self._selecionar_guided_candidates(scan, frame_data)
        tempo_scan = float(time.perf_counter() - inicio_scan)
        guided_full_scores, guided_full_errors, tempo_guided_full = self._executar_guided_full(guided_full_frames, full_cache)
        guided_full_values = [x['effort_score'] for x in guided_full_scores]
        guided_full = self._resumo_branch(guided_full_values, self.guided_max_frames)
        guided_face_scores, guided_face_errors, tempo_guided_face = self._executar_guided_face(selected_candidates)
        guided_face_values = [x['effort_face_score'] for x in guided_face_scores]
        guided_face = self._resumo_branch(guided_face_values, self.guided_max_frames)
        usa_checkpoint_facial = self._usa_checkpoint_facial()
        decisao_primaria = guided_face if usa_checkpoint_facial else baseline
        scores_primarios = guided_face_scores if usa_checkpoint_facial else baseline_scores
        erros_primarios = guided_face_errors if usa_checkpoint_facial else baseline_errors
        fonte_decisao = 'EFFORT_SCENE_FACE_GUIDED_FACE_CROP' if usa_checkpoint_facial else 'EFFORT_UNIFORM_FULL_TOP25_MEAN'
        if decisao_primaria['score'] is None:
            result = self._resultado_sem_score('SEM_SCORE_EFFORT_PRIMARIO_VALIDO')
            result['classification_reason'] = 'SEM_SCORE_EFFORT_PRIMARIO_VALIDO'
            result['effort_errors'] = erros_primarios
            result['guided_selection_mode'] = selection_mode
            result['guided_selection'] = {
                'source_frames': int(scan['source_frames']),
                'scan_frames': int(scan['scan_frames']),
                'scene_count': int(scan['scene_count']),
                'faces_detected': int(scan['faces_detected']),
                'selected_face_frames': int(len(selected_candidates))
            }
            result['decision_policy'] = {'primary_decision': fonte_decisao}
            result['runtime_seconds'].update({
                'effort_baseline': float(tempo_baseline),
                'face_scene_scan': float(tempo_scan),
                'effort_guided_full': float(tempo_guided_full),
                'effort_guided_face': float(tempo_guided_face),
                'effort': float(tempo_baseline + tempo_guided_full + tempo_guided_face),
                'visual_total': float(time.perf_counter() - inicio_total)
            })
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open('w', encoding='utf-8') as arquivo:
                json.dump(result, arquivo, ensure_ascii=False, indent=4)
            return result
        tempo_effort = tempo_baseline + tempo_guided_full + tempo_guided_face
        tempo_total = float(time.perf_counter() - inicio_total)
        result = {
            'visual_score': float(decisao_primaria['score']),
            'visual_score_method': self.SCORE_METHOD,
            'visual_screening_method': self.SCREENING_METHOD,
            'classification': decisao_primaria['classification'],
            'classification_reason': decisao_primaria['classification_reason'],
            'visual_confidence': float(decisao_primaria['confidence']),
            'classification_confidence': float(decisao_primaria['confidence']),
            'visual_consistency': float(decisao_primaria['consistency']),
            'aggregation_method': self.AGGREGATION_METHOD,
            'fusion_reason': 'EFFORT_DUALVIEW_SEM_FUSAO',
            'consistency_gate_applied': bool(decisao_primaria['score'] >= self.limite_suspeito and decisao_primaria['consistency'] < self.consistencia_minima_suspeito),
            'suspicious_consistency_min': float(self.consistencia_minima_suspeito),
            'threshold': float(self._limiar_binario_effort()),
            'decision_thresholds': {
                'nao_suspeito_max': float(self.limite_nao_suspeito),
                'suspeito_min': float(self.limite_suspeito),
                'limiar_binario': float(self._limiar_binario_effort()),
                'consistencia_minima': float(self.consistencia_minima_suspeito)
            },
            'decision_policy': {
                'primary_decision': fonte_decisao,
                'experimental_guided_full': 'EFFORT_SCENE_FACE_GUIDED_FULL',
                'experimental_guided_face': 'EFFORT_SCENE_FACE_GUIDED_FACE_CROP',
                'mode': 'FACE_CHECKPOINT_CROP_PRIMARY' if usa_checkpoint_facial else 'GENIMAGE_FULL_FRAME_PRIMARY',
                'views_fused': False,
                'note': 'O checkpoint facial usa guided-face como decisao primaria.' if usa_checkpoint_facial else 'Guided-full e guided-face sao screenings; a decisao usa o baseline uniforme.'
            },
            'score_effort': float(decisao_primaria['score']),
            'score_effort_mean': float(decisao_primaria['statistics']['media']),
            'score_effort_top25_mean': float(decisao_primaria['score']),
            'classification_effort': decisao_primaria['classification'],
            'classification_reason_effort': decisao_primaria['classification_reason'],
            'confidence_effort': float(decisao_primaria['confidence']),
            'consistency_effort': float(decisao_primaria['consistency']),
            'effort_statistics': decisao_primaria['statistics'],
            'frame_scores_effort': scores_primarios,
            'effort_errors': erros_primarios,
            'score_effort_guided_full': guided_full['score'],
            'classification_effort_guided_full': guided_full['classification'],
            'classification_reason_effort_guided_full': guided_full['classification_reason'],
            'confidence_effort_guided_full': guided_full['confidence'],
            'consistency_effort_guided_full': guided_full['consistency'],
            'effort_guided_full_statistics': guided_full['statistics'],
            'frame_scores_effort_guided_full': guided_full_scores,
            'effort_guided_full_errors': guided_full_errors,
            'score_effort_guided_face': guided_face['score'],
            'classification_effort_guided_face': guided_face['classification'],
            'classification_reason_effort_guided_face': guided_face['classification_reason'],
            'confidence_effort_guided_face': guided_face['confidence'],
            'consistency_effort_guided_face': guided_face['consistency'],
            'effort_guided_face_statistics': guided_face['statistics'],
            'frame_scores_effort_guided_face': guided_face_scores,
            'effort_guided_face_errors': guided_face_errors,
            'guided_selection_mode': selection_mode,
            'guided_selection': {
                'source_frames': int(scan['source_frames']),
                'scan_frames': int(scan['scan_frames']),
                'scan_max_frames': int(self.guided_scan_max_frames),
                'scene_count': int(scan['scene_count']),
                'faces_detected': int(scan['faces_detected']),
                'face_detection_rate': float(scan['face_detection_rate']),
                'valid_face_candidates': int(len(scan['candidates'])),
                'selected_face_frames': int(len(selected_candidates)),
                'guided_full_frames': int(len(guided_full_frames)),
                'guided_max_frames': int(self.guided_max_frames),
                'face_margin': float(self.guided_face_margin),
                'min_face_side': int(self.guided_min_face_side),
                'scene_bhattacharyya_min': float(self.scene_bhattacharyya_min),
                'scene_mad_min': float(self.scene_mad_min),
                'selected': selected_candidates,
                'scan_records': scan['scan_records']
            },
            'yunet_model': {
                'path': str(self.yunet_model_path),
                'source': self.YUNET_DOWNLOAD_URL,
                'score_threshold': float(self.yunet_score_threshold),
                'nms_threshold': float(self.yunet_nms_threshold),
                'role': 'APENAS_SELECAO_E_CROP; NAO_CLASSIFICA_DEEPFAKE'
            },
            'effort_model': {
                'name': 'Effort / CLIP ViT-L/14',
                'training_checkpoint': self.effort_weights_path.stem,
                'checkpoint_path': str(self.effort_weights_path),
                'clip_model': self.effort_clip_model_name,
                'fake_class_index': 1
            },
            'score_sbi': None,
            'classification_sbi': None,
            'confidence_sbi': 0.0,
            'sbi_influencia_decisao': False,
            'score_commfor': None,
            'classification_commfor': None,
            'confidence_commfor': 0.0,
            'score_npr': None,
            'classification_npr': None,
            'frame_scores': scores_primarios,
            'selected_frame_scores': scores_primarios,
            'frames_analyzed': int(len(scores_primarios)),
            'faces_analyzed': int(len(guided_face_scores)),
            'visual_statistics': {
                'decision_source': fonte_decisao,
                'primary': decisao_primaria['statistics'],
                'baseline': baseline['statistics'],
                'guided_full': guided_full['statistics'],
                'guided_face': guided_face['statistics']
            },
            'runtime_seconds': {
                'effort_baseline': float(tempo_baseline),
                'face_scene_scan': float(tempo_scan),
                'effort_guided_full': float(tempo_guided_full),
                'effort_guided_face': float(tempo_guided_face),
                'effort': float(tempo_effort),
                'visual_total': float(tempo_total)
            }
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open('w', encoding='utf-8') as arquivo:
            json.dump(result, arquivo, ensure_ascii=False, indent=4)
        print('-' * 76)
        print('Resumo visual - EFFORT DUAL VIEW / SEM FUSAO')
        print(f"Decisao primaria ({fonte_decisao}): {decisao_primaria['score']:.4f} | {decisao_primaria['classification']}")
        if guided_full['score'] is None:
            print('Guided-full: sem score')
        else:
            print(f"Guided-full: {guided_full['score']:.4f} | {guided_full['classification']} | n={len(guided_full_scores)}")
        if guided_face['score'] is None:
            print('Guided-face: sem score facial valido')
        else:
            print(f"Guided-face: {guided_face['score']:.4f} | {guided_face['classification']} | n={len(guided_face_scores)}")
        print(f"Selecao: scenes={scan['scene_count']} | faces={scan['faces_detected']}/{scan['scan_frames']} | selecionados={len(selected_candidates)}")
        print('Nenhuma fusao foi aplicada entre as leituras visuais.')
        print(f'Tempos: baseline={tempo_baseline:.2f}s | scan={tempo_scan:.2f}s | guided-full={tempo_guided_full:.2f}s | guided-face={tempo_guided_face:.2f}s | visual total={tempo_total:.2f}s')
        return result

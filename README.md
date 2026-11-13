# Detector Multimodal de Deepfakes Audiovisuais

O **Detector Multimodal de Deepfakes Audiovisuais** é uma aplicação desenvolvida em Python para identificar indícios de geração ou manipulação artificial em vídeos. O sistema analisa **imagens e áudio**, combina os indicadores por meio de **fusão multimodal** e classifica cada vídeo como **`SUSPEITO`** ou **`NAO_SUSPEITO`**.

A pasta principal do código é **`V3_Final/`**. Esta é a **V3 Final**, utilizada como versão principal do projeto. As versões anteriores (V1 e V2) representam os estágios experimentais descritos no trabalho de conclusão de curso e podem ser disponibilizadas nas Releases do repositório.

> **Importante:** as classificações são indicadores computacionais de suspeita. Elas não constituem comprovação definitiva de que uma mídia foi manipulada.

## Como funciona

1. **Entrada:** o programa identifica automaticamente um ou vários vídeos dentro da pasta `input/` e suas subpastas.
2. **Pré-processamento:** `FFprobe` lê os metadados; `FFmpeg` extrai o áudio em WAV mono de 16 kHz; `OpenCV` extrai quadros para análise.
3. **Análise visual em cascata:** `RapidOCR` procura marcas textuais de geração artificial; `SPSL` examina características forenses; `Effort` com `CLIP ViT-L/14` analisa as imagens, utilizando `YuNet` como apoio na localização de rostos. A cascata pode dispensar etapas posteriores quando seus critérios forem atendidos.
4. **Análise sonora:** `DF-Arena-1B`, executado com `ONNX Runtime`, analisa segmentos da faixa sonora para identificar indícios de fala sintética.
5. **Fusão multimodal:** reúne os indicadores dos dois ramos e produz a classificação final. Também é possível executar somente a modalidade visual ou sonora.
6. **Resultados:** salva scores, classificações, evidências, tempos e relatórios em **CSV, JSON e PDF** na pasta `output/`.

A aplicação oferece execução em lote, perfis de uso de CPU e retomada de processamento interrompido. Na V3, o uso do RapidOCR pode ser configurado no modo multimodal.

## Requisitos

- **Windows 10/11** ou **Linux** (exemplos abaixo para Ubuntu 24.04 e Debian 12).
- **Python 3.11 ou 3.12**.
- **FFmpeg e FFprobe** disponíveis no sistema.
- Internet para instalar bibliotecas e, se necessário, carregar arquivos auxiliares de CLIP/RapidOCR na primeira execução.
- Espaço em disco e memória RAM suficientes para os modelos, bibliotecas e vídeos (a instalação pode ocupar vários gigabytes).

Os **modelos** e os **datasets utilizados nos experimentos** são disponibilizados separadamente por meio do Google Drive, para evitar que o repositório fique excessivamente grande. Baixe os arquivos e coloque-os nas pastas indicadas na seção seguinte **antes de iniciar a aplicação**.

## Download dos modelos e datasets (Google Drive)

Os arquivos necessários à execução e à reprodução dos testes não acompanham o código-fonte do GitHub. Eles devem ser obtidos pelos links abaixo.

| Conteúdo | Link para download |
|---|---|
| **Modelos e checkpoints** (SPSL, Effort, YuNet e DF-Arena-1B) | [Baixar modelos e checkpoints](https://drive.google.com/file/d/1Kai0aXR3SepXI2NjRiNBgrdGnmbrdlKH/view?usp=sharing) |
| **Datasets utilizados** (vídeos de avaliação) | [Baixar datasets](https://drive.google.com/file/d/1UQtI1MqClK-_x3_PU-o5SaxSuim7StkC/view?usp=sharing) |

> **Acesso:** confira se os links do Google Drive estão compartilhados como **“Qualquer pessoa com o link — Leitor”** para permitir o download.

**Depois de baixar:**

1. Extraia os arquivos dos modelos, caso estejam compactados, para a pasta **`V3_Final/models/`**, respeitando as subpastas `audio/` e `visual/` e os nomes originais dos checkpoints.
2. Extraia os datasets para **`V3_Final/input/`**. Quando os dados tiverem rótulos conhecidos, preserve a separação em `REAL/` e `FAKE/` e as subpastas de idiomas utilizadas na avaliação.
3. Confira se a estrutura ficou semelhante à apresentada abaixo. Evite pastas duplicadas, como `models/models/` ou `input/input/`, pois o programa procura os arquivos nos caminhos configurados.

Os arquivos auxiliares do CLIP e do RapidOCR poderão exigir acesso à internet na primeira execução. Os datasets são necessários para reproduzir os experimentos, mas você também pode colocar seus próprios vídeos em `input/` para realizar análises.

## Instalação e primeira execução

Abra a pasta **`V3_Final/`** no **VS Code** e abra seu terminal integrado (`Terminal` → `Novo Terminal`). Execute **somente o bloco do seu sistema operacional**; não há scripts de instalação adicionais.

### Windows — PowerShell

```powershell
winget install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements
winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Se Python ou FFmpeg acabarem de ser instalados, pode ser necessário **reabrir o terminal do VS Code** para que os comandos sejam reconhecidos. Nesse caso, execute novamente apenas as linhas a partir de `py -3.11 -m venv .venv`. Os comandos usam o PyTorch para **CPU**.

### Linux — Ubuntu 24.04 / Debian 12

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ffmpeg libsndfile1
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python main.py
```

Antes da instalação, confira se `python3 --version` mostra Python **3.11 ou 3.12**. Em outras distribuições Linux, os nomes dos pacotes de sistema podem ser diferentes.

### Bibliotecas Python

As dependências Python estão reunidas no `requirements.txt` e são instaladas pelos comandos da seção anterior, sem necessidade de instalar cada biblioteca separadamente.

Antes de iniciar o programa, confira se os checkpoints baixados do **Google Drive** estão em `models/` e se os vídeos estão em `input/`. Nenhum script de instalação ou download adicional é necessário. Alguns arquivos auxiliares do CLIP e do RapidOCR podem ser obtidos automaticamente na primeira execução, caso ainda não estejam em cache.

## Estrutura do projeto

Após baixar e extrair os modelos e datasets do Google Drive, a organização esperada é:

```text
V3_Final/
├── README.md
├── requirements.txt
├── main.py
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py   # FFprobe, FFmpeg e OpenCV
│   ├── visual.py          # RapidOCR, SPSL, YuNet e Effort
│   ├── audio.py           # DF-Arena-1B
│   ├── fusion.py          # Fusão multimodal
│   └── execution.py       # Recursos, tempos e métricas
├── models/
│   ├── audio/
│   │   └── df_arena_1b_int8.onnx
│   └── visual/
│       ├── spsl_best.pth
│       ├── face_detection_yunet_2023mar.onnx
│       ├── effort_clip_L14_trainOn_FaceForensic.pth
│       └── huggingface/  # Arquivos auxiliares/cache do CLIP
├── input/
│   ├── REAL/
│   │   └── en/
│   │       └── exemplo_real.mp4
│   └── FAKE/
│       └── en/
│           └── exemplo_fake.mp4
└── output/                # Criado durante o processamento
```

**Modelo Effort:** se o checkpoint facial não estiver disponível, o código também aceita `models/visual/effort_clip_L14_trainOn_sdv14.pth`. Para reproduzir os experimentos, utilize os **mesmos checkpoints** adotados nos testes originais.

**Datasets:** coloque os vídeos obtidos no Google Drive diretamente em `input/` e suas subpastas. A separação em `REAL/` e `FAKE/` é opcional para análise, mas necessária para que o sistema reconheça os rótulos de referência e calcule métricas de avaliação. O nome do arquivo não é usado como rótulo de previsão. Evite caracteres acentuados nos caminhos dos arquivos no Windows, pois algumas operações de gravação de quadros do OpenCV podem falhar nesses caminhos.

A pasta `output/` é gerada pelo programa e não precisa acompanhar o repositório.

## Como usar depois da instalação

Para iniciar novamente, execute **somente**:

**Windows:**

```powershell
.\.venv\Scripts\python.exe main.py
```

**Linux:**

```bash
./.venv/bin/python main.py
```

No menu, escolha a modalidade **Visual**, **Sonora** ou **Multimodal** e o perfil de desempenho desejado. O programa informa a quantidade de vídeos encontrados em `input/` antes do processamento. Há opções para iniciar a análise e **continuar uma execução interrompida**.

## Arquivos gerados e avaliação

Os arquivos de saída são organizados em `output/`, conforme a modalidade e a configuração utilizada. Entre eles estão:

- CSV com classificações, scores e informações por vídeo;
- JSON com o resumo do processamento;
- relatório final em PDF com as classificações, etapas executadas e tempos;
- evidências visuais e registros de desempenho, quando disponíveis.

Quando a entrada possui pastas com rótulos `REAL` e `FAKE`, o relatório inclui métricas como **matriz de confusão, acurácia, precisão, sensibilidade, especificidade, F1 e acurácia balanceada**. Algumas métricas podem ficar indisponíveis quando o conjunto contém somente uma classe.

## Solução de problemas

| Problema | O que verificar |
|---|---|
| Python não encontrado | Instale Python 3.11/3.12 e reabra o terminal. |
| `ffmpeg` ou `ffprobe` não encontrado | Verifique `ffmpeg -version` e `ffprobe -version` em um terminal novo. |
| `ModuleNotFoundError` | Confirme que o ambiente `.venv` foi criado e que `requirements.txt` foi instalado. |
| Checkpoint ausente | Baixe o pacote de modelos pelo link do Google Drive e confira o nome e o caminho do arquivo em `models/`. |
| Falha na inicialização do CLIP ou OCR | Confira a conexão à internet e a disponibilidade dos arquivos de cache necessários. |
| Quadros não salvos | Use um caminho sem acentos/caracteres especiais e confirme permissão de gravação. |
| Falta de memória | Experimente o perfil econômico ou um número menor de vídeos. |
| Dataset não encontrado | Baixe o conjunto pelo link do Google Drive e extraia os vídeos para `input/`, sem criar uma pasta `input/input/`. |
| Resultados diferentes dos experimentos | Confira os modelos, versões das bibliotecas, parâmetros, modalidade e arquivos de entrada. |

## Histórico das versões

- **V1:** configuração inicial do pipeline de detecção de deepfakes multimodal.
- **V2:** refinamento dos critérios de decisão.
- **V3 Final:** versão principal, com ajustes finais, retomada de execução e RapidOCR configurável no modo multimodal.

As versões anteriores podem ser disponibilizadas nas **Releases** do repositório, enquanto a página principal apresenta a V3.

## Uso acadêmico

Este projeto foi desenvolvido para pesquisa e avaliação experimental. Seus resultados devem ser interpretados como **apoio à triagem**, não como prova definitiva de manipulação. Respeite as licenças e as condições de uso dos modelos e datasets disponibilizados nos links externos.

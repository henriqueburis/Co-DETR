# VOXAR_temp
# pode se verifica as libs e os resultados no google colab do projeto VOXAR_lab_alt.ipynb.
#  Video Demo Doc

Este README fornece documentação e instruções para usar o código de demonstração de vídeo do MMDetection. O código é projetado para detecção de objetos em vídeos usando modelos pré-treinados do MMDetection, uma popular caixa de ferramentas de detecção de objetos.

**Tabela de Conteúdos**
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Argumentos](#argumentos)
- [Saída](#saída)
- [Exemplo](#exemplo)
- [Licença](#licença)

## Pré-requisitos

Antes de usar este código, certifique-se de ter os seguintes pré-requisitos:

- Python 3.6 ou superior
- PyTorch (consulte o [guia de instalação do PyTorch](https://pytorch.org/get-started/locally/) para instalação)
- MMDetection (consulte o [guia de instalação do MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md) para instalação)
- OpenCV (`opencv-python`)
- mmcv (biblioteca de visão computacional personalizada da MMLab)

## Instalação

1. Clone o repositório MMDetection, se ainda não o tiver feito:

   ```bash
   !git clone https://github.com/Sense-X/Co-DETR.git
   cd Co-DETR


1. Instale o MMDetection e suas dependências:

   ```bash
    !pip install '/content/Co-DETR'
   out Successfully installed mmdet-2.25.3 terminaltables-3.1.10


1. Instale os pacotes Python necessários:

   ```bash
    pip install mmcv opencv-python

Coloque o código fornecido no diretório MMDetection/demo ou crie um novo arquivo Python com o código.

## Uso
Para usar a demonstração de vídeo do MMDetection, siga estas etapas:


1. Abra um terminal.

1. Execute o script de demonstração de vídeo com o seguinte comando:

   ```bash
    python video_demo_step1.py <arquivo_de_vídeo> <arquivo_de_configuração> <arquivo_de_checkpoint> [opções]
    python video_demo_step2.py <arquivo_de_vídeo> <arquivo_de_configuração> <arquivo_de_checkpoint> [opções]
    python video_demo_step3.py <arquivo_de_vídeo> <arquivo_de_configuração> <arquivo_de_checkpoint> [opções]

Substitua <arquivo_de_vídeo>, <arquivo_de_configuração> e <arquivo_de_checkpoint> pelos caminhos do seu arquivo de vídeo, arquivo de configuração do MMDetection e arquivo de ponto de verificação do modelo, respectivamente.

## Argumentos
Aqui estão os argumentos de linha de comando disponíveis para o script de demonstração de vídeo:

video: Caminho para o arquivo de vídeo de entrada.
config: Caminho para o arquivo de configuração do MMDetection.
checkpoint: Caminho para o arquivo de ponto de verificação do modelo.
--device: Especifique o dispositivo usado para inferência (padrão: 'cuda:0').
--score-thr: Limiar de pontuação da caixa delimitadora (padrão: 0,3).
--out: Especifique o arquivo de vídeo de saída.
--show: Exiba o vídeo durante o processamento.
--wait-time: O intervalo (em segundos) para exibir quadros (padrão: 1).

## Saída
Se você especificar a opção --out, o vídeo processado será salvo no arquivo especificado.
Se você especificar a opção --show, o vídeo processado será exibido em uma janela.

## Exemplo

1. Aqui está um exemplo de como executar a demonstração de vídeo:

   ```bash
      python video_demo.py video_de_entrada.mp4 configs/arquivo_de_configuração.py checkpoints/modelo_de_checkpoint.pth --out video_de_saída.mp4 --show

Este comando processará video_de_entrada.mp4 usando os arquivos de configuração e ponto de verificação especificados, salvará o resultado em video_de_saída.mp4 e exibirá o vídeo durante o processamento.

Licença
Buris - Este código é protegido por direitos autorais da OpenMMLab. Para detalhes sobre a licença, consulte a licença fornecida pela OpenMMLab.

# TCC - Detecção de Exercícios com IA

Este projeto faz parte do Trabalho de Conclusão de Curso (TCC) e tem como objetivo utilizar técnicas de Visão Computacional e Redes Neurais para avaliar a execução de exercícios físicos.

## 📌 Tecnologias Utilizadas

- Python 3
- MediaPipe
- TensorFlow / Keras
- OpenCV
- NumPy
- Git & GitHub

## 📁 Estrutura do Projeto

```bash
TCC CODIGO/
├── Deteccao_Exercicios_AI-master/   # Código-fonte do projeto
├── videos para testar/              # Vídeos utilizados para validação
├── README.md                        # Este arquivo
└── .gitignore                       # Arquivos ignorados pelo Git
```
## Como executar o projeto

1. Baixe e instale o Python 3.11 pelo site oficial: https://www.python.org/downloads/release/python-3110/  
   Durante a instalação, **marque a opção "Add Python to PATH"**.

2. Abra o terminal e navegue até a pasta do projeto com os comandos abaixo. A pasta raiz se chama `tcc codigo`.

```bash
cd "tcc codigo"
cd Deteccao_Exercicios_AI-master
cd mediapipe_lstm
```

3. Instale as bibliotecas necessárias com o comando:

```bash
pip install opencv-python mediapipe tensorflow numpy
```

4. Com tudo instalado, execute o script principal para selecionar o vídeo e validar a execução:

```bash
python analyze_video.py
```

O sistema abrirá uma janela para você escolher o vídeo. Ele será processado automaticamente e validado com base no modelo treinado.

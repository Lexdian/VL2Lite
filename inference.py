import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os

# -------------------------------------------------------------------------
# AJUSTE ESTRUTURAL
# -------------------------------------------------------------------------
try:
    # Tenta adicionar o diretório raiz do projeto ('VL2Lite') ao Python path
    # Isso é necessário para que a importação 'from src.models...' funcione
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
except:
    pass

# IMPORTAÇÃO CORRIGIDA: Usa o caminho confirmado (src.models.kd_module.KDModule)
try:
    from src.models.kd_module import KDModule as VL2LiteModel 
except ImportError as e:
    print("ERRO: Falha ao importar KDModule. Verifique o ambiente virtual ou a estrutura de pastas.")
    sys.exit(1)


def load_cub_classes(path):
    """Carrega as 200 classes do CUB-200-2011 na ordem correta."""
    classes = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # O formato é: "ID Nome_da_Classe" (ex: 1 001.Black_footed_Albatross)
                # Queremos o nome formatado para leitura (ex: "Black footed Albatross")
                parts = line.strip().split(' ', 1)
                if len(parts) > 1:
                    # Remove a parte "001." e substitui '_' por espaço
                    name = parts[1].split('.', 1)[-1].replace('_', ' ')
                    classes.append(name)
        return classes
    except FileNotFoundError:
        print(f"ERRO: Arquivo de classes não encontrado em {path}. Usando placeholder.")
        return [f"Espécie {i+1}" for i in range(200)]


def run_inference():
    """
    Carrega o modelo VL2Lite (KDModule) treinado e executa a inferência.
    """
    
    # ====================================================================
    # ⚠️ CAMPOS OBRIGATÓRIOS: SUBSTITUIR APENAS O CAMINHO DA IMAGEM
    #
    
    # 1. Caminho para o checkpoint (SEU CKPT)
    CKPT_PATH = r"D:\GitHub\VL2Lite\logs\train\runs\2025-10-29_00-14-24\checkpoints\epoch_029.ckpt"
    
    # 2. Caminho para o arquivo classes.txt
    CLASSES_FILE_PATH = r"D:\GitHub\VL2Lite\data\kd_datasets\0_CUB_200_2011\classes.txt"
    
    # 3. Caminho para a imagem que você deseja testar
    # 💡 SUBSTITUIR PELO CAMINHO REAL DA SUA IMAGEM
    IMAGE_PATH = r"D:\GitHub\VL2Lite\Images_Test\RedHeaded.jpeg" 
    
    # ====================================================================
    
    # 4. Carrega a lista de classes
    CLASS_NAMES = load_cub_classes(CLASSES_FILE_PATH)
    
    # 5. Configuração do Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 6. Carregar o Modelo
    try:
        print(f"Carregando modelo a partir de: {CKPT_PATH}")
        model = VL2LiteModel.load_from_checkpoint(
            checkpoint_path=CKPT_PATH, 
            map_location=device 
        )
        model.eval() 
        model.to(device)
    except Exception as e:
        print(f"\n--- ERRO CRÍTICO AO CARREGAR CHECKPOINT ---")
        print(f"Detalhe: {e}")
        return

    # 7. Transformações (Padrão para modelos de visão 224x224)
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 8. Preparar a Imagem de Entrada
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"ERRO ao abrir imagem em {IMAGE_PATH}. Detalhe: {e}")
        return

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 9. Executar a Inferência
    print("\nExecutando inferência...")
    with torch.no_grad():
        output = model(image_tensor)
        
        # 💥 CORREÇÃO PRINCIPAL: Extrair os logits da tupla de saída.
        # Baseado em kd_module.py, os logits de classificação estão no índice 1 da tupla.
        if isinstance(output, tuple):
            # Se for uma tupla (features, logits), pegue o segundo elemento (logits)
            logits = output[1]
            print("AVISO: Saída do modelo foi uma tupla, usando o índice 1 como logits.")
        else:
            # Se for um Tensor (caso improvável, mas seguro), use-o diretamente
            logits = output 

    probabilities = torch.softmax(logits, dim=1)
    predicted_score, predicted_idx = torch.max(probabilities, 1)

    # 10. Apresentar os Resultados
    idx = predicted_idx.item()
    score = predicted_score.item()
    
    print("\n--- RESULTADOS DA CLASSIFICAÇÃO ---")
    print(f"Conjunto de Dados: CUB-200-2011 ({len(CLASS_NAMES)} espécies)")
    print(f"Índice da Classe Prevista: {idx}")
    print(f"Confiança: {score:.4f}")
    
    if idx < len(CLASS_NAMES):
        print(f"Espécie Prevista: {CLASS_NAMES[idx]}")
    else:
        print("Erro de índice: O modelo retornou uma classe inválida.")


if __name__ == "__main__":
    run_inference()
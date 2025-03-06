import argparse
import torch
import torch.nn as nn
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights
from PIL import Image
import torchvision.transforms as transforms

def load_model(weights_path, num_classes, device):
    """
    Crea la arquitectura del modelo y carga los pesos entrenados.
    Se utiliza la versión preentrenada por defecto para inicializar,
    y se modifica la última capa para obtener 13 clases.
    """
    # Usamos los pesos de imagenet para inicializar la arquitectura
    weights = MNASNet1_0_Weights.IMAGENET1K_V1
    model = mnasnet1_0(weights=weights)
    # Modificar la última capa para que tenga las salidas deseadas
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Cargar los pesos entrenados
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def predict_image(model_species, model_behavs, image_path, device):
    """
    Carga la imagen, aplica las transformaciones requeridas y
    realiza la inferencia, retornando el índice de la clase predicha.
    """
    # Se usan las transformaciones por defecto de MNASNet
    transform = MNASNet1_0_Weights.DEFAULT.transforms()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Agrega dimensión de batch
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs_sp = model_species(input_tensor)
        outputs_beh = model_behavs(input_tensor)
    _, predicted_sp = torch.max(outputs_sp, 1)
    _, predicted_beh = torch.max(outputs_beh, 1)
    return predicted_sp.item(), predicted_beh.item()

def main():
    parser = argparse.ArgumentParser(description='Script de inferencia con modelo MNASNet entrenado')
    parser.add_argument('--image', type=str, required=True,
                        help='Ruta a la imagen para realizar la inferencia.')
    parser.add_argument('--weights-sp', type=str, default='best_model.pt',
                        help='Ruta al archivo de pesos del modelo clasificador de especies.')
    parser.add_argument('--weights-beh', type=str, default='best_model.pt',
                        help='Ruta al archivo de pesos del modelo clasificador de comportamientos.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Dispositivo a utilizar ("cuda" o "cpu").')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f'Usando dispositivo: {device}')
    
    model_species = load_model(args.weights_sp, 13, device)
    model_behavs = load_model(args.weights_beh, 4, device)
    predicted_sp, predicted_beh = predict_image(model_species, model_behavs, args.image, device)
    print(f'La especie predicha es: {predicted_sp}')
    print(f'El comportamiento predicho es: {predicted_beh}')

if __name__ == '__main__':
    main()

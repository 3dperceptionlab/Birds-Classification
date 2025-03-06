import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, MobileNet_V3_Large_Weights, mobilenet_v3_large
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import copy
import os
import wandb
from tqdm import tqdm
from dataset import BirdsDataset
from PIL import Image
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=10, device='cuda'):
    """
    Función para entrenar el modelo.
    En cada época se entrena y se valida el modelo.
    Durante la validación se calculan precisión, recall y F1,
    y se registran en wandb. Se implementa early stopping basado en F1,
    guardando el modelo cada vez que se mejora esta métrica.
    Además, se utiliza tqdm para mostrar el progreso de cada fase.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    epochs_no_improve = 0

    since = time.time()

    for epoch in range(num_epochs):
        print(f'Época {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            phase_iter = tqdm(dataloaders[phase], desc=f'{phase.capitalize()}', leave=False)
            for inputs, labels in phase_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                phase_iter.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'val':
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                f1 = f1_score(all_labels, all_preds, average='weighted')
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

                wandb.log({
                    'epoch': epoch,
                    'val_loss': epoch_loss,
                    'val_precision': precision,
                    'val_recall': recall,
                    'val_f1': f1,
                })

                scheduler.step(f1)

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'weights/best_model.pt')
                    print("Nuevo mejor modelo guardado.")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')
                wandb.log({
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                })

        if epochs_no_improve >= patience:
            print(f'Early stopping activado en la época {epoch+1}')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Entrenamiento completado en {int(time_elapsed//60)}m {int(time_elapsed % 60)}s')
    print(f'Mejor F1 obtenido: {best_f1:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device='cuda'):
    """
    Función para evaluar el modelo sobre el conjunto de validación.
    Se calculan la pérdida, precisión, recall y F1.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluación'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    loss_total = running_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Evaluación - Loss: {loss_total:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento o evaluación de modelo MNASNet para clasificación de aves.')
    parser.add_argument('--only_eval', action='store_true',
                        help='Ejecuta solo evaluación sin entrenamiento.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas para entrenamiento.')
    parser.add_argument('--weights', type=str, default=None,
                        help='Ruta a los pesos del modelo a cargar. Si no se proporcionan, se usan los pesos por defecto.')
    parser.add_argument('--model', type=str, default=None,
                        help='Arquitectura del modelo.')
    args = parser.parse_args()

    # Inicializar wandb y configurar parámetros del experimento
    wandb.init(project="species-clf", name=args.model, config={
        "epochs": args.epochs,
        "batch_size": 64,
        "learning_rate": 0.001,
        "patience": 10,
        "model": args.model
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Usando dispositivo: {device}')

    # Definir las transformaciones para entrenamiento y validación utilizando las transformaciones por defecto de MNASNet
    data_transforms = {
        'train': MNASNet1_0_Weights.DEFAULT.transforms(),
        'val': MNASNet1_0_Weights.DEFAULT.transforms(),
    }

    # Se asume que los datos se encuentran en la carpeta "data" con subdirectorios "train" y "val"
    train_dataset = BirdsDataset(csv_file='./data/train_bounding_boxes.csv', frames_dir='/data/frames', transform=data_transforms['train'])
    val_dataset = BirdsDataset(csv_file='./data/test_bounding_boxes.csv', frames_dir='/data/frames', transform=data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Crear el modelo
    num_classes = 13
    if args.model == 'mnasnet':
        weights = MNASNet1_0_Weights.DEFAULT
        model = mnasnet1_0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif args.model == 'efficientnet':
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif args.model == 'mobilenet':
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError('Modelo no válido. Debes especificar un modelo válido (mnasnet, efficientnet o mobilenet).')
    # Cargar pesos si se especificaron
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, weights_only=True))
        print(f'Pesos cargados desde {args.weights}')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # Monitorear el modelo con wandb
    #wandb.watch(model, criterion, log="all", log_freq=100)

    if args.only_eval:
        print("Modo solo evaluación activado.")
        evaluate_model(model, val_loader, criterion, device=device)
    else:
        model = train_model(model, dataloaders, criterion, optimizer, scheduler,
                            num_epochs=config.epochs, patience=config.patience, device=device)

    wandb.finish()

if __name__ == '__main__':
    main()

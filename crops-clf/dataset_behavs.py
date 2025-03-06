import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import MNASNet1_0_Weights

class BirdsDataset(Dataset):
    def __init__(self, csv_file, frames_dir, transform=None):
        """
        Args:
            csv_file (str): Ruta al archivo CSV con las anotaciones.
            frames_dir (str): Ruta al directorio "frames" donde se encuentran las imágenes.
            transform (callable, optional): Transformaciones a aplicar al recorte.
        
        Se espera que el CSV tenga las columnas: 
          species_id;species;video_name;frame;bounding_boxes
        El campo bounding_boxes debe ser una cadena con una o varias bounding boxes separadas por '|'.
        Cada bounding box debe tener el formato "x1,y1,x2,y2".
        """
        # Cargar las anotaciones con pandas (se especifica ";" como separador)
        self.annotations = pd.read_csv(csv_file, sep=";")
        self.frames_dir = frames_dir
        self.transform = transform
        self.samples = []  # Lista donde se almacenará cada muestra (una por cada bounding box)
        
        # Iterar sobre cada fila del CSV
        for idx, row in self.annotations.iterrows():
            species_id = row['species_id']
            species = row['species']
            video_name = row['video_name']
            frame = row['frame']
            bboxes_str = row['bounding_boxes']
            bboxes_str = bboxes_str.strip("[(").strip(')]')
            
            # Construir la ruta completa a la imagen:
            # Se asume que la imagen se encuentra en frames/<video_name>/<frame>
            image_path = os.path.join(frames_dir, str(video_name), "frame_" + str(frame).zfill(5) + ".jpg")
            
            # Si hay más de una bounding box, se separan con "|" (en caso de una sola, se obtiene una lista de un elemento)
            if isinstance(bboxes_str, str) and bboxes_str.strip() != "":
                bbox_list = bboxes_str.split("), (")
            else:
                bbox_list = []
            
            for bbox_str in bbox_list:
                coords = bbox_str.split(",")
                if len(coords) != 6:
                    # Si no se tienen 6 valores, se omite esta bounding box
                    print('ERROR: Incorrect bounding box format')
                    continue
                try:
                    # Convertir cada coordenada a entero (puedes usar float si es necesario)
                    bbox = tuple(map(float, coords))
                except ValueError:
                    print(tuple(map(float, coords)))
                    print('ERROR: Incorrect bounding box value format')
                    continue  # Omitir la bounding box si hay problemas en la conversión
                
                # Agregar un sample por cada bounding box encontrada
                self.samples.append({
                    'image_path': image_path,
                    'bbox': bbox
                })
        print(f"Se encontraron {len(self.samples)} bounding boxes en {len(self.annotations)} imágenes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        bbox = sample['bbox'][:4]
        
        # Abrir la imagen y asegurar que esté en formato RGB
        image = Image.open(image_path)
        
        # Realizar el recorte de la imagen usando la bounding box: (left, upper, right, lower)
        crop = image.crop(bbox)
        
        # Aplicar las transformaciones si se proporcionaron
        if self.transform:
            crop = self.transform(crop)
        
        # Devolver el recorte y la etiqueta (species_id)
        return crop, int(sample['bbox'][4])

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
#from torchvision.models import mnasnet1_0, MNASNet1_0_Weights
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, MobileNet_V3_Large_Weights, mobilenet_v3_large
from PIL import Image
import torch.nn as nn
import time
import select, tty, termios
import curses

def load_model_clf(weights_path, num_classes, device, model_name):
    """
    Crea la arquitectura del modelo y carga los pesos entrenados.
    Se utiliza la versión preentrenada por defecto para inicializar,
    y se modifica la última capa para obtener 13 clases.
    """
    # Usamos los pesos de imagenet para inicializar la arquitectura
    #weights = MNASNet1_0_Weights.IMAGENET1K_V1
    if model_name == 'mnasnet':
        model = mnasnet1_0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet':
        model = efficientnet_v2_s()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        model = mobilenet_v3_large()
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    # Modificar la última capa para que tenga las salidas deseadas
    
    
    # Cargar los pesos entrenados
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def opencv_to_torch(img):
    """
    Converts an OpenCV image (NumPy array) to a PyTorch tensor.
    Args:
        img (numpy.ndarray): OpenCV image in BGR format.
    Returns:
        torch.Tensor: PyTorch tensor with shape [C, H, W].
    """
    # Convert BGR to RGB (PyTorch expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy array to PyTorch tensor
    tensor = torch.from_numpy(img)
    
    # Move channel dimension to the first position (HWC -> CHW)
    tensor = tensor.permute(2, 0, 1)  # Shape: [C, H, W]
    
    # Convert to float and normalize
    tensor = tensor.float() / 255.0
    
    return tensor

def predict_image_clf(model_species, model_behavs, image, device):
    """
    Carga la imagen, aplica las transformaciones requeridas y
    realiza la inferencia, retornando el índice de la clase predicha.
    """
    # Se usan las transformaciones por defecto de MNASNet
    #transform = MNASNet1_0_Weights.DEFAULT.transforms()
    transform = MobileNet_V3_Large_Weights.DEFAULT.transforms()
    #image = Image.open(image_path).convert('RGB')
    input_tensor = opencv_to_torch(image)
    input_tensor = transform(input_tensor).unsqueeze(0)  # Agrega dimensión de batch
    #input_tensor = input_tensor.unsqueeze(0)  # Agrega dimensión de batch
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs_sp = model_species(input_tensor)
        outputs_beh = model_behavs(input_tensor)
        print(torch.softmax(outputs_sp[0],0))
        print(torch.softmax(outputs_beh[0],0))
    _, predicted_sp = torch.max(outputs_sp, 1)
    _, predicted_beh = torch.max(outputs_beh, 1)
    return predicted_sp.item(), predicted_beh.item()

@smart_inference_mode()
def run(
        stdscr,
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.70,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        weights_sp='best_model.pt', # Weights of species classification model
        weights_beh='best_model.pt', # Weights of behavior classification model
        clf=False    # Predict species/behavior after YOLO detection
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        #view_img = check_imshow(warn=True) # DEV: Comment line to avoid xcb error
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    annot_file = './annotations/annotations.csv'
    exists_file = os.path.exists(annot_file)
    out_annotations_file = open(annot_file, 'a')
    if not exists_file:
        out_annotations_file.write('image_name,species_detected,behavior_detected,bbox\n')
        out_annotations_file.close()
    stdscr.nodelay(1)
    stdscr.addstr('Press "q" key to exit from the execution\n')
    for path, im, im0s, vid_cap, s in dataset:
        key = stdscr.getch()
        if key == ord('q'):
            stdscr.addstr('"q" key pressed. Exiting...\n')
            stdscr.refresh()
            break
        out_annotations_file = open(annot_file, 'a')
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        species_preds, behavs_preds = [], []
        for i, det in enumerate(pred):  # per image
            det = det[det[:, -1] == 14]
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            im_save = im0.copy()
            #imc = im0.copy() if save_crop else im0  # for save_crop
            imc = im0.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            crops = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        crop = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    else:
                        crop = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True, save=False) 
                    crops.append(crop)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            # Save annotations
            if len(det):
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                cv2.imwrite(f'./annotations/frame_{timestamp}.jpg', im_save)

            # Classification
            #LOGGER.info('### Classification results ###')
            if clf:
                for crop in crops:
                    model_species = load_model_clf(weights_sp, 13, device, 'mobilenet')
                    model_behavs = load_model_clf(weights_beh, 4, device, 'mobilenet')
                    predicted_sp, predicted_beh = predict_image_clf(model_species, model_behavs, crop, device) # im0
                    cv2.imwrite(f'./annotations/frame_crop.jpg', crop)
                    species_preds.append(predicted_sp)
                    behavs_preds.append(predicted_beh)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        if clf:
            for s,b in zip(species_preds, behavs_preds):
                LOGGER.info(f'La especie predicha es: {s}')
                LOGGER.info(f'El comportamiento predicho es: {b}\n')
                #import pdb; pdb.set_trace()
                if len(det):
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                    out_annotations_file.write(f'{timestamp}.jpg,{s},{b},{det[:][0:5]}\n')
        else:
            if len(det):
                out_annotations_file.write(f'{timestamp}.jpg,x,x,{det[:][0:5]}\n')

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    out_annotations_file.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--weights-sp', type=str, default='best_model.pt', help='Ruta al archivo de pesos del modelo clasificador de especies.')
    parser.add_argument('--weights-beh', type=str, default='best_model.pt', help='Ruta al archivo de pesos del modelo clasificador de comportamientos.')
    parser.add_argument('--clf', action='store_true', help='classify detections')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(stdscr,opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(stdscr=stdscr,**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    #main(opt)
    curses.wrapper(main, opt)

import pandas as pd
import json
import argparse

def split_bounding_boxes(
    bounding_boxes_csv: str = "bounding_boxes_2.csv",
    splits_json: str = "splits.json"
):
    # 1. Cargar el archivo CSV
    df = pd.read_csv(bounding_boxes_csv, sep=';')
    
    # 2. Cargar splits desde splits.json
    with open(splits_json, "r") as f:
        splits = json.load(f)
    
    # Extraer cada lista de videos
    train_videos = splits["train_set"]
    test_videos = splits["test_set"]
    val_videos = splits["val_set"]
    
    # 3. Filtrar según el split
    train_df = df[df["video_name"].isin(train_videos)]
    test_df = df[df["video_name"].isin(test_videos)]
    val_df = df[df["video_name"].isin(val_videos)]
    
    # 4. Guardar los resultados en CSV
    train_df.to_csv("train_bounding_boxes.csv", sep=';', index=False)
    test_df.to_csv("test_bounding_boxes.csv", sep=';', index=False)
    val_df.to_csv("val_bounding_boxes.csv", sep=';', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide bounding boxes según splits (train, test, val).")
    parser.add_argument(
        "--bounding_boxes",
        type=str,
        default="bounding_boxes_2.csv",
        help="Ruta al archivo bounding_boxes.csv"
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default="splits.json",
        help="Ruta al archivo splits.json"
    )
    
    args = parser.parse_args()
    
    split_bounding_boxes(
        bounding_boxes_csv=args.bounding_boxes,
        splits_json=args.splits_json
    )

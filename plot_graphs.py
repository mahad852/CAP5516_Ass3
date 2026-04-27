import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    
    return parser.parse_args()

def plot_graph(data, xtitle, ytitle, title, path):
    plt.cla()

    epochs = len(data)
    plt.plot(x=epochs, y=data)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.title(title)

    plt.savefig(path)

def main():
    args = get_args()

    folds = range(5)

    avg_loss = []
    avg_dice = []
    avg_aji = []
    avg_pq = []

    total_folds = 0.0

    for fold in folds:
        fold_dir = os.path.join(args.model_dir, str(fold))

        if not os.path.exists(fold_dir):
            continue
        
        logs_path = os.path.join(fold_dir, "logs.json")
        if not os.path.exists(logs_path):
            continue

        with open(logs_path, "r") as f:
            logs = json.load(f)

        train_losses = logs["train_losses"]
        val_metrics = logs["val_metrics"]

        if len(avg_loss) == 0:
            avg_loss = np.zeros(shape=(len(train_losses)), dtype=float)
            avg_pq = np.zeros(shape=(len(train_losses)), dtype=float)
            avg_dice = np.zeros(shape=(len(train_losses)), dtype=float)
            avg_aji = np.zeros(shape=(len(train_losses)), dtype=float)

        avg_loss = np.asarray(train_losses) + avg_loss

        dice = np.asarray([m["dice"] for m in val_metrics])
        aji = np.asarray([m["aji"] for m in val_metrics])
        pq = np.asarray([m["pq"] for m in val_metrics])

        avg_dice = avg_dice + dice
        avg_aji = avg_aji + aji
        avg_pq = avg_pq + pq

        total_folds += 1

    avg_loss /= total_folds
    avg_dice /= total_folds
    avg_aji /= total_folds
    avg_pq /= total_folds

    
    plot_graph(data=avg_loss, xtitle="Epochs", ytitle="Average Training Loss", title="Average Train Loss v. epochs", path=os.path.join(args.model_dir, "train_loss.png"))
    plot_graph(data=avg_dice, xtitle="Epochs", ytitle="Average Dice", title="Average Dice v. epochs", path=os.path.join(args.model_dir, "dice.png"))
    plot_graph(data=avg_dice, xtitle="Epochs", ytitle="Average AJI", title="Average AJI v. epochs", path=os.path.join(args.model_dir, "aji.png"))
    plot_graph(data=avg_dice, xtitle="Epochs", ytitle="Average PQ", title="Average PQ v. epochs", path=os.path.join(args.model_dir, "pq.png"))

if __name__ == "__main__":
    main()
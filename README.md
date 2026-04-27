# CAP5516 Assignment 3 - Nuclei Segmentation #

## Installation ##
```
conda create -n cap5516_ass3 python=3.11
conda activate cap5516_ass3
pip install -r requirements.txt
```

## Training ##
To train for the 5 folds, run this command:
```
python main.py \
 --root <dataset_root> \
 --seed <seed for reproducibility, 42 by default> \
 --sam_path <local path for the pretrained SAM vit-base if you have one, otherwise huggingface_hub would be used to fetch>
 --batch_size <16 by default>
 --epochs <10 by default>
 --output_dir <where fold logs and weights are stored, they'll be stored inside output_dir/<fold_id>>
```

## Evaluation ##
For evaluation, make sure you use the same seed as the training. Run this:
```
python eval.py \
 --root <dataset_root> \
 --seed <seed for reproducibility, 42 by default> \
 --sam_path <local path for the pretrained SAM vit-base if you have one, otherwise huggingface_hub would be used to fetch> \
 --output_dir <dir where sample image and pred masks would be stored> \
 --fold <fold_id> \
 --approach <either point or auto> \
 --lora_path <path of the LoRA model traiend during the previous step. make sure the path is for the right fold>
```

## Plotting average graphs ##
To plot the average (train loss, dice, AJI, PQ) over all the folds, run:
```
python plot_graphs.py --model_dir <same as the output_dir passed to main.py>
```

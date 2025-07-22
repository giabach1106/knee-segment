```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

```bash
uv run wandb login  
```

```bash
uv run python train.py all                    
uv run python visualize.py      
```

```bash
uv run python train.py single --name baseline
uv run python train.py single --name clahe
uv run python train.py single --name crop_fixed
uv run python train.py list
```


| Experiment | description | preprocessing |
|------------|-------------|---------------|
| `baseline` | No preprocessing | None |
| `clahe` | Contrast enhancement | CLAHE |
| `crop_fixed` | ROI extraction | Fixed crop |
| `snake_crop` | Adaptive ROI | Snake algorithm |
| `hist_eq` | Global contrast | Histogram equalization |
| `blur` | Noise reduction | Gaussian blur |
| `sharpen` | Edge enhancement | Unsharp masking |
| `normalize` | Intensity scaling | Min-max normalization |
| `augmentation` | Data augmentation | Flips, rotations, elastic |
| `optimizer_sgd` | Different optimizer | SGD vs Adam |
| `multichannel` | Multi-channel input | Original + CLAHE |
| `combined_clahe_crop_aug` | Best combination | CLAHE + Crop + Augmentation |


### Metrics
- *Dice Coefficient*
- *IoU*
- *Pixel Accuracy*


### Dependencies
```bash
# Clean reinstall
rm -rf .venv
uv sync
```


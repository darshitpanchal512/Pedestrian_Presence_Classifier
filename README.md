# Pedestrian-Presence-Classifier-
A lightweight binary CNN classifier for autonomous vehicle safety systems, optimized cascaded detection pipelines by acting as an efficient pre-filter, reducing overall safety system load.
Why bother with something so simple? Because in a real autonomous vehicle, running heavy detection models on every frame is expensive. A cheap "gate-keeper" like this one can quickly clear the empty scenes, so the heavy models only run when they're actually needed. That saves computing power without cutting corners on safety.

---

## Results at a glance

The final (fine-tuned) model was tested on 74 validation images it never saw during training.

| Metric | Score | What it means in plain terms |
|---|---|---|
| **Accuracy** | **90.54%** | How often the model was right overall |
| **Recall (Sensitivity)** | **100%** | It caught **every single pedestrian** — zero were missed |
| **Precision** | **84.44%** | When it said "pedestrian," it was right about 5 out of 6 times |
| **Specificity** | **80.55%** | How well it correctly cleared empty scenes |

**Confusion matrix (validation set):**

|  | Predicted: No Pedestrian | Predicted: Pedestrian |
|---|---|---|
| **Actual: No Pedestrian** | 29 (correct) | 7 (false alarm) |
| **Actual: Pedestrian** | 0 (missed) | 38 (correct) |

For a safety system, missing a pedestrian is the worst possible mistake. A few false alarms are a much safer trade-off — the car just double-checks a scene that turned out to be empty.

---

## The dataset

The images come from the **Penn-Fudan Database** (sourced via Kaggle).

| Group | Count |
|---|---|
| FudanPed (with pedestrians) | 74 |
| PennPed (with pedestrians) | 96 |
| Background (no pedestrians) | 199 |
| **Total** | **369** |

The data was split **80/20** using a fixed random seed (42) so the split is reproducible:

- **Training:** 295 images
- **Validation:** 74 images
- Both sets are kept roughly balanced between the two classes.

### Expected folder layout

The code uses PyTorch's `ImageFolder`, which reads the class label from the folder name. Set your data up like this:

```
Datasets_Project/
├── Pedestrian/        # images that contain people
├── No Pedestrian/     # background / empty road scenes
└── Test/              # held-out images for final inference only
```

The `Test` folder is automatically excluded from training and validation. It's used only at the very end to see how the model behaves on brand-new images.

---

## How the model is built

It's a deliberately simple network — three convolutional blocks followed by a small classifier head.

```
Input image (224 x 224 x 3)
        │
   Conv Block 1  → 32 channels → ReLU → MaxPool   (224 → 112)
   Conv Block 2  → 64 channels → ReLU → MaxPool   (112 → 56)
   Conv Block 3  → 128 channels → ReLU → MaxPool  (56 → 28)
        │
   Flatten → Fully Connected (512) → ReLU → Dropout (0.4)
        │
   Fully Connected (1) → Sigmoid
        │
   Output: probability of "Pedestrian" (0.0 to 1.0)
```

A few notes on the design choices:

- **Convolutional blocks** are the part that actually "sees." The early blocks pick up simple things like edges and colors; the later ones combine those into more human-shaped patterns.
- **Max pooling** shrinks the image down at each step (224 → 112 → 56 → 28). It keeps the important stuff and throws away the noise, which also keeps the model fast.
- **Dropout (0.4)** randomly switches off 40% of the neurons during training. With only 369 images, the model could easily just memorize them instead of learning. Dropout forces it to learn general patterns instead.
- **Sigmoid output** squeezes the final answer into a number between 0 and 1. Above 0.5 means "Pedestrian"; below means "No Pedestrian."

---

## Training setup

| Setting | Value |
|---|---|
| Loss function | Binary Cross-Entropy (`BCELoss`) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 16 |
| Max epochs | 20 |
| Early stopping | Patience of 5 (stops if validation accuracy doesn't improve for 5 epochs) |

**Data augmentation (training only):** each training image is randomly flipped, rotated a little (±10°), and has its brightness/contrast/color slightly shifted. This is a cheap trick to make a small dataset feel bigger — the model sees the same pedestrian in slightly different conditions each time, so it learns to recognize them more flexibly. Validation images are left untouched so the score reflects real performance.

---

## Why these settings? (Hyperparameter tuning)

Rather than guessing, each key setting was tested to see which value worked best. Here's what the experiments showed:

**Learning rate** — controls how big a step the model takes when it learns.

| Value | Validation Accuracy |
|---|---|
| 0.01 | 48.65% (too big — the model couldn't settle) |
| **0.001** | **89.19% ✅ chosen** |
| 0.0001 | 89.19% |
| 0.00001 | 87.84% |

**Batch size** — how many images the model looks at before updating.

| Value | Validation Accuracy |
|---|---|
| 8 | 89.19% |
| **16** | **89.19% ✅ chosen** |
| 32 | 88.44% |

**Dropout** — how many neurons to switch off during training.

| Value | Validation Accuracy |
|---|---|
| **0.4** | **90.54% ✅ chosen** |
| 0.5 | 89.19% |
| 0.6 | 89.19% |

The takeaway: a learning rate of 0.001 that's too high (0.01) completely breaks the model, while dropout of 0.4 gave the extra push that took accuracy over 90%.

---

## Getting started

### 1. Requirements

Create a file called `requirements.txt` with the following, or just install these packages directly:

```
torch
torchvision
matplotlib
numpy
scikit-learn
seaborn
Pillow
```

Install them:

```bash
pip install -r requirements.txt
```

### 2. Get the data

Download the Penn-Fudan images and arrange them into the folder layout shown above (`Pedestrian/`, `No Pedestrian/`, `Test/`).

### 3. Point the code at your data

The script was originally written for Google Colab, so the dataset path is set near the top. Update it to wherever your data lives:

```python
DATA_PATH = '/content/drive/MyDrive/Datasets_Project'   # <- change this to your path
```

### 4. Train the model

Run the main script (or open the notebook and run the cells top to bottom):

```bash
python cnn_pedestrian_detection_model.py
```

Training will:
- Train for up to 20 epochs, stopping early if it stops improving
- Save the best-performing model as `best_pedestrian_model.pth`
- Save two plots: `training_curves.png` (loss and accuracy over time) and `confusion_matrix.png`

### 5. Test on new images

The script includes a `predict_image()` function that takes any image and returns a label plus a confidence score. Drop new images into your `Test/` folder and the script will run through them, printing results like:

```
street_01.jpg        -> No Pedestrian ( 99.6 %)
crossing_04.jpg      -> Pedestrian    ( 94.2 %)
```

---

## How it behaves on real, unseen images

The model was tried on fresh dashboard-camera photos it had never seen. Here's the honest picture:

- **Empty streets — excellent.** It clears "No Pedestrian" scenes with about **99.3%** average confidence. It rarely cries wolf.
- **Clear, close pedestrians — strong.** When one or two people are plainly visible, confidence sits around **94–96%**.
- **Crowded or partly hidden pedestrians — shaky.** When people are small in the frame, overlapping, or half-hidden behind cars and poles, confidence can drop to around **66%**. This is the model's weak spot.

---

## Limitations

Being upfront about what this model *can't* do is part of using it responsibly:

- **Struggles with occlusion.** People partly hidden behind objects are much harder for it, and that's exactly what happens in real traffic.
- **Struggles with busy scenes.** Lots of cars, trees, poles, and shadows make it harder for the network to pick out the person.
- **It only says yes or no.** There are no bounding boxes, no locations, and no head-count. A real driver-assistance system needs to know *where* the pedestrian is, not just *that* there is one.
- **Only tested in fair conditions.** It hasn't been checked on night driving, rain, fog, motion blur, or unusual camera angles.
- **Possible dataset bias.** If the training images lean toward clear, easy pedestrian shots, the model may quietly be biased toward simpler cases.

---

## Where this could go next (Note: I would love to connect with you if you are optimizing same model)

- **A bigger, smarter architecture** to handle small and crowded pedestrians better.
- **Occlusion-aware training** — deliberately hide parts of pedestrians during training (techniques like Cutout or Random Erasing) so the model learns to cope with it.
- **Upgrade to object detection** — move from "is there a pedestrian?" to "where is each pedestrian?" with bounding boxes, which is what real ADAS systems need.
- **Use video, not single frames** — checking several frames in a row would confirm a pedestrian and cut down on false alarms.
- **Test in tough conditions** — systematically measure performance at night, in fog and rain, and from different angles, then report results per condition.


---

## Acknowledgments

- **Dataset:** [Penn-Fudan Database for Pedestrian Detection](https://www.cis.upenn.edu/~jshi/ped_html/) (accessed via Kaggle).
- **Author:** Darshitkumar Panchal
- Developed as a project at **Technische Hochschule Ingolstadt**, Faculty of Electrical Engineering and Information Technology.

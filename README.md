# Algorithm for classifying lung nodules in chest CT

The source code for the algorithm container for classifying 3D blocks of lung nodules from chest CT (shaped `[128, 128, 64]` in x, y, z), generated with evalutils version 0.3.1.

This algorithm contains the inference scripts for VGG16 models that estimate the malignancy risk and predict the nodule type of lung nodules. 

This GitHub repository is linked to the [baseline algorithm](https://grand-challenge.org/algorithms/luna22-ismi-baseline/) for the [LUNA22-ISMI](https://luna22-ismi.grand-challenge.org/) challenge. 

## Interfaces

### Inputs

- CT image (`/inputs/images/ct/<uuid>.mha`)

### Outputs

- Lung nodule malignancy risk (`/outputs/lung-nodule-malignancy-risk.json`)
- Lung nodule type (`/outputs/lung-nodule-type.json`)

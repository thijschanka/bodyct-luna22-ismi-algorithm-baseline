from typing import Dict

import SimpleITK
import numpy as np

import tensorflow.keras
from tensorflow.keras.applications import VGG16

# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")
from data import (
    load_and_resample_nodule_img,
    center_crop_volume,
    get_cross_slices_from_cube,
)

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


def clip_and_scale(
    data: np.ndarray,
    min_value: float = -1000.0,
    max_value: float = 400.0,
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data


class Nodule_classifier(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self.input_size = 224
        self.input_spacing = 0.2

        # load malignancy model
        self.model_malignancy = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=2,
            classifier_activation="softmax",
        )
        self.model_malignancy.load_weights(
            "/opt/algorithm/models/vgg16_malignancy_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )

        # load texture model
        self.model_texture = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
            classifier_activation="softmax",
        )
        self.model_texture.load_weights(
            "/opt/algorithm/models/vgg16_noduletype_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )

        print("Models initialized")

    def preprocess(
        self,
        img: SimpleITK.Image,
    ) -> SimpleITK.Image:

        # Resample image
        original_spacing_mm = img.GetSpacing()
        original_size = img.GetSize()
        new_spacing = (self.input_spacing, self.input_spacing, self.input_spacing)
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(
                original_size,
                original_spacing_mm,
                new_spacing,
            )
        ]
        resampled_img = SimpleITK.Resample(
            img,
            new_size,
            SimpleITK.Transform(),
            SimpleITK.sitkLinear,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )

        # Return image data as a numpy array
        return SimpleITK.GetArrayFromImage(resampled_img)

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:

        print(f"Processing image of size: {input_image.GetSize()}")

        nodule_data = self.preprocess(input_image)
        # Crop a volume of 50 mm^3 around the nodule
        nodule_data = center_crop_volume(
            volume=nodule_data,
            crop_size=np.array(
                (
                    self.input_size,
                    self.input_size,
                    self.input_size,
                )
            ),
            pad_if_too_small=True,
            pad_value=-1024,
        )

        # Extract the axial/coronal/sagittal center slices of the 50 mm^3 cube
        nodule_data = get_cross_slices_from_cube(volume=nodule_data)
        nodule_data = clip_and_scale(nodule_data)

        malignancy = self.model_malignancy(nodule_data[None]).numpy()[0, 1]
        texture = np.argmax(self.model_texture(nodule_data[None]).numpy())

        result = dict(
            malignancy_risk=str(np.round(malignancy, 3)),
            texture=str(texture),
        )

        print(result)

        return result


if __name__ == "__main__":
    Nodule_classifier().process()

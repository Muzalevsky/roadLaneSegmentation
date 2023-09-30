import albumentations as albu
import cv2
import numpy as np


class LaneMarkingAlbuAugmentation:
    def __init__(self):
        ssr_params = dict(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=15,
            interpolation=3,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        )

        cd_params = dict(
            min_holes=100,
            max_holes=150,
            min_height=10,
            max_height=50,
            min_width=10,
            max_width=50,
            p=0.5,
            mask_fill_value=0,
        )

        self.description = [
            albu.ToGray(p=0.3),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf(
                [
                    albu.GaussNoise(p=0.5),  # no effect
                    albu.MultiplicativeNoise(per_channel=True, p=0.3),  # no effect
                ],
                p=0.4,
            ),
            albu.ImageCompression(
                quality_lower=10, quality_upper=20, p=0.5
            ),  # jpg? almost no effect
            albu.OneOf(
                [
                    albu.MotionBlur(blur_limit=33, p=0.2),
                    albu.MedianBlur(blur_limit=23, p=0.2),
                    albu.GaussianBlur(blur_limit=27, p=0.2),
                    albu.Blur(blur_limit=13, p=0.2),
                ],
                p=0.2,
            ),
            albu.OneOf(
                [
                    albu.CLAHE(),
                    albu.Sharpen(),
                    albu.Emboss(),
                    albu.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
            albu.HueSaturationValue(p=0.3),
            albu.OneOf(
                [
                    # Black background
                    albu.ShiftScaleRotate(
                        **ssr_params,
                        value=(0, 0, 0),
                    ),
                    # White background
                    albu.ShiftScaleRotate(
                        **ssr_params,
                        value=(255, 255, 255),
                    ),
                ]
            ),
            albu.OneOf(
                [
                    albu.CoarseDropout(
                        **cd_params,
                        fill_value=(0, 0, 0),
                    ),
                    albu.CoarseDropout(
                        **cd_params,
                        fill_value=(255, 255, 255),
                    ),
                ]
            ),
        ]

        self.compose = albu.Compose(self.description, p=1)

    def __call__(self, img, mask):
        transformed = self.compose(image=img, mask=mask)
        mask = np.array(transformed["mask"])
        img = transformed["image"]
        return img, mask

    def serialize(self):
        return albu.to_dict(self.compose)

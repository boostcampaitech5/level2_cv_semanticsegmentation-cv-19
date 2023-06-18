import albumentations as A


class BaseAugmentation(object):
    def __init__(self, img_size, is_train):
        self.is_train = is_train
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __call__(self, image, label=None):
        inputs = {"image": image}
        if label is not None:
            inputs["mask"] = label

        if self.transforms is not None:
            result = self.transforms(**inputs)
            image = result["image"]
            if label is not None:
                label = result["mask"]

        return image, label

    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )


class CustomAugmentation(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.OneOf(
                        [
                            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
                        ],
                        p=0.9,
                    ),
                    A.OneOf([A.Blur(blur_limit=3, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.1),
                    A.CLAHE(clip_limit=(1, 4), p=1),

                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )

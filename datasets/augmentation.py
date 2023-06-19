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
                    A.RandomResizedCrop(512, 512, scale=(0.2, 1.0), ratio=(0.4, 1.0), always_apply=False, p=1.0),
                    # A.HorizontalFlip(p=0.5),
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
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    # A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
                    A.ElasticTransform(alpha=15.0, sigma=2.0),
                    A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.CLAHE(clip_limit=(1, 4), p=0.6),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )


class CustomAugmentation1(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.05, p=0.6),
                    A.ElasticTransform(alpha=15.0, sigma=2.0, alpha_affine=25),
                    A.JpegCompression(quality_lower=85, quality_upper=95, p=0.1),
                    A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.CLAHE(clip_limit=(1, 4), p=0.5),
                    A.Cutout(num_holes=10, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )

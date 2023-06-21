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
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    # A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
                    A.ElasticTransform(alpha=15.0, sigma=2.0),
                    A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.HorizontalFlip(p=0.5),
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
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    # A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
                    # A.ElasticTransform(alpha=15.0, sigma=2.0),
                    A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.CLAHE(clip_limit=(1, 4), p=0.4),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )


class CustomAugmentation2(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    # A.Resize(self.img_size, self.img_size),
                    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
                    # A.ElasticTransform(alpha=15.0, sigma=2.0),
                    # A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    # A.HorizontalFlip(p=0.5),
                    # A.CLAHE(clip_limit=(1, 4), p=0.4),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(1024, 1024),
                ]
            )


class TestAugmentation(BaseAugmentation):
    def __init__(self, img_size, is_train):
        self.is_train = is_train
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __call__(self, image):
        inputs = {"image": image}
        if self.transforms is not None:
            result = self.transforms(**inputs)
            image = result["image"]

        return image

    def get_transforms(self):
        return A.Compose(
            [
                A.Resize(2048, 2048),  # A.Resize(self.img_size, self.img_size),
            ]
        )


class CustomAugmentationElastic(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [A.Resize(self.img_size, self.img_size), A.ElasticTransform(p=1, alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03)]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )


class CustomAugmentation1_elastic(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    # A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
                    A.ElasticTransform(p=1, alpha=60, sigma=60 * 0.10, alpha_affine=60 * 0.06),
                    A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.CLAHE(clip_limit=(1, 4), p=0.4),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )


class CustomAugmentation1_nonelastic(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    # A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
                    # A.ElasticTransform(p=1, alpha=60, sigma=60 * 0.10, alpha_affine=60 * 0.06,
                    A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.CLAHE(clip_limit=(1, 4), p=0.4),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )


class CustomAugmentationElastic_9733all(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.ElasticTransform(p=1, alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                ]
            )

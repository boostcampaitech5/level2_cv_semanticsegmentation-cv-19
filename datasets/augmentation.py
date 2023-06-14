import albumentations as A

BaseAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

CustomAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

TestAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

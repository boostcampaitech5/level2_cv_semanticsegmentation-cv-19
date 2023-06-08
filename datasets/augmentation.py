import albumentations as A

BaseAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

CustomAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

TestAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

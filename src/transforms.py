import albumentations as albu
from albumentations.pytorch import ToTensorV2


class ImageTransform:
    def __init__(self, img_size=224):
        self.transform = {
            'train': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0),

            'test': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0)
        }

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        augmented = augmented['image']

        return augmented

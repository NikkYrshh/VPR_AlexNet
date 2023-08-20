import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class VPRClass(object):

    """
    VPR class for feature extraction with AlexNet pytorch
        model: AlexNetConv3
    """

    def __init__(self):
        """
        Initializes the VPR class with a truncated version of the AlexNet

        AlexNetConv3  contains first three convolutional layers of the full AlexNet model
        """
        from torchvision.models.alexnet import AlexNet_Weights
        # original_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # Load the AlexNet model with default pre-trained weights
        original_model = models.alexnet(weights=AlexNet_Weights.DEFAULT)

        # original_model = models.alexnet(pretrained=True) <-- old version now has been deprecated

        # Truncated AlexNet model 3 layers
        class AlexNetConv3(nn.Module):
            def __init__(self):
                super(AlexNetConv3, self).__init__()
                self.features = nn.Sequential(
                    *list(original_model.features.children())[:7]  # stop at 7
                )

            # Forward pass to extract features
            def forward(self, x):
                x = self.features(x)
                return x

        self.model = AlexNetConv3()
        self.model.eval()

        # Resizing, normalising, converting to tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Batch feature extraction
    def extract_features_batch(self, image_paths):
        images = [Image.open(p) for p in image_paths]
        image_tensors = torch.stack([self.transform(img) for img in images])
        with torch.no_grad():
            features = self.model(image_tensors)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return features

import torch
import torch.nn as nn
import torchvision.models as models


class VehicleClassifier(nn.Module):
    # Dictionary mapping model names to their configurations
    MODEL_CONFIGS = {
        # ResNet models
        "resnet18": {
            "builder": models.resnet18,
            "weights": models.ResNet18_Weights.DEFAULT,
            "feature_dim": 512,
        },
        "resnet34": {
            "builder": models.resnet34,
            "weights": models.ResNet34_Weights.DEFAULT,
            "feature_dim": 512,
        },
        "resnet50": {
            "builder": models.resnet50,
            "weights": models.ResNet50_Weights.DEFAULT,
            "feature_dim": 2048,
        },
        "resnet101": {
            "builder": models.resnet101,
            "weights": models.ResNet101_Weights.DEFAULT,
            "feature_dim": 2048,
        },
        # EfficientNet models
        "efficientnet_b0": {
            "builder": models.efficientnet_b0,
            "weights": models.EfficientNet_B0_Weights.DEFAULT,
            "feature_dim": 1280,
        },
        "efficientnet_b1": {
            "builder": models.efficientnet_b1,
            "weights": models.EfficientNet_B1_Weights.DEFAULT,
            "feature_dim": 1280,
        },
        "efficientnet_b2": {
            "builder": models.efficientnet_b2,
            "weights": models.EfficientNet_B2_Weights.DEFAULT,
            "feature_dim": 1408,
        },
        # MobileNet models
        "mobilenet_v3_small": {
            "builder": models.mobilenet_v3_small,
            "weights": models.MobileNet_V3_Small_Weights.DEFAULT,
            "feature_dim": 576,
        },
    }

    # Layers to freeze for different model types
    FREEZE_CONFIGS = {
        "resnet": ["conv1", "bn1", "layer1", "layer2"],
        "efficientnet": None,  # Special handling for EfficientNet
        "mobilenet": None,  # Special handling for MobileNet
    }

    def __init__(
        self,
        num_classes,
        pretrained=True,
        freeze_layers=False,
        dropout_rate=0.5,
        model_name="resnet50",
    ):
        super(VehicleClassifier, self).__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {list(self.MODEL_CONFIGS.keys())}"
            )

        # Get model configuration
        config = self.MODEL_CONFIGS[model_name]

        # Initialize backbone with explicit float32 dtype
        self.original_model = config["builder"](
            weights=config["weights"] if pretrained else None
        )

        # Explicitly convert all model parameters to float32
        for param in self.original_model.parameters():
            param.data = param.data.float()

        feature_dim = config["feature_dim"]

        # Remove the original classifier based on model type
        if model_name.startswith("resnet"):
            self.pool = nn.Identity()  # ResNets already have pooling
            self.backbone = nn.Sequential(
                *list(self.original_model.children())[:-1]
            )  # Remove FC layer
        else:
            # For EfficientNet and MobileNet
            self.backbone = nn.Sequential(
                *list(self.original_model.children())[:-1]
            )
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze layers if specified
        if freeze_layers:
            self._freeze_layers(model_name)

        # New classifier head with regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Make sure all parameters are float32
        for param in self.parameters():
            param.data = param.data.float()

    def _freeze_layers(self, model_name):
        """Freeze early layers based on model type"""
        # Get model family (resnet, efficientnet, etc.)
        model_family = next(
            (
                family
                for family in self.FREEZE_CONFIGS
                if model_name.startswith(family)
            ),
            None,
        )

        if model_family == "resnet":
            layers_to_freeze = self.FREEZE_CONFIGS[model_family]
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False
        elif model_family in ["efficientnet", "mobilenet"]:
            print(f"Skipping layer freezing for {model_family} models.")
            # Freeze the first half of the features
            # features = self.backbone[0].features
            # total_blocks = len(features)
            # for i, block in enumerate(features):
            #     if i < total_blocks // 2:
            #         for param in block.parameters():
            #             param.requires_grad = False

    def forward(self, x):
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.float()

        features = self.backbone(x)
        pooled = self.pool(features)
        return self.classifier(pooled)

    def unfreeze_layers(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True

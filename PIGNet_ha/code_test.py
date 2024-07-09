import torch
from swin  import SwinTransformer

# Define Swin Transformer model
net = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=3,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)

# Create a dummy input tensor with shape (7, 3, 513, 513)
dummy_x = torch.randn(7, 3, 513, 513)

# Forward pass through the network
logits = net(dummy_x)  # Output shape will be (7, 3) due to batch size 7 and num_classes 3

# Print model summary and logits
print(net)
print(logits)

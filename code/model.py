from torch import nn

class PKLot_modelV1(nn.Module):
  def __init__(self, input_channels : int, hidden_units : int, output_channels : int):
    super().__init__()
    self.cnn_block_1 = nn.Sequential(
        nn.Conv2d(
            in_channels = input_channels,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = hidden_units,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                         stride=2)
    )
    self.cnn_block_2 = nn.Sequential(
        nn.Conv2d(
            in_channels = hidden_units,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = hidden_units,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 2560,
                  out_features = output_channels)
    )

  def forward(self, x):
    x = self.cnn_block_1(x)
    x = self.cnn_block_2(x)
    x = self.classifier(x)
    return x
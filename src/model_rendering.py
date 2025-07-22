import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import inspect
import os

# ==== Definição dos modelos ====

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=4)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.gn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2, padding=2)
        self.gn3 = nn.GroupNorm(8, 32)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.gn4 = nn.GroupNorm(8, 64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.gn5 = nn.GroupNorm(8, 64)
        self.conv6 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.gn6 = nn.GroupNorm(8, 64)
        self.fc1 = nn.Linear(1024, 128)
        self.gn7 = nn.GroupNorm(8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.gn4(self.conv4(x)))
        x = F.relu(self.gn5(self.conv5(x)))
        x = F.relu(self.gn6(self.conv6(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.gn7(self.fc1(x))))

        return F.log_softmax(self.fc2(x), dim=1)

# ==== Geração dos grafos ====

output_dir = "./model_graphs"
os.makedirs(output_dir, exist_ok=True)

# Identifica todas as classes que herdam de nn.Module e que foram definidas no script
model_classes = [
    obj for name, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj.__module__ == "__main__"
]

# Dummy input compatível com MNIST
dummy_input = torch.randn(1, 1, 28, 28)

for model_class in model_classes:
    model = model_class()
    model.eval()  # Evita comportamentos de dropout/batchnorm durante o forward
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    output_path = os.path.join(output_dir, f"{model_class.__name__}")
    dot.format = "svg"
    dot.render(output_path)
    print(f"Gráfico do modelo '{model_class.__name__}' salvo em: {output_path}")
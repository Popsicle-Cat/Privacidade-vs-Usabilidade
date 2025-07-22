import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import time
from tqdm import tqdm

# Configuração
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado para treinamento: {device}")
epochs = 30
delta = 1e-5
file_path = "resultados_pytorch.txt"

# Preprocessamento e carregamento de dados
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_data = train_dataset.data.unsqueeze(1).float() / 255.0
train_labels = train_dataset.targets
test_data = test_dataset.data.unsqueeze(1).float() / 255.0
test_labels = test_dataset.targets

# Modelos
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

# Treinamento
def train_model(model, optimizer, criterion, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for X_batch, y_batch in epoch_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

# Avaliação
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted')
    rec = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return acc, prec, rec, f1

models = {'a': ModelA, 'b': ModelB}
learning_rates = [0.01, 0.005, 0.001]
batch_sizes = [25, 50, 100]
l2_norm_clips = [0.5, 1.0, 1.5]
noise_multipliers = [0.5, 1.0, 1.5]
num_microbatches_options = [5, 10]

test_number = 0

with open(file_path, "w") as f:
    f.write("Resultados dos testes de modelos DP e não DP\n")
    f.write(f"Todos os testes foram executados com: delta = {delta} e epochs = {epochs}\n\n")

for model_key, lr, batch_size in itertools.product(models.keys(), learning_rates, batch_sizes):

    for dp in [False, True]:

        if dp:
            for l2_norm_clip, noise_multiplier, num_microbatches in itertools.product(l2_norm_clips, noise_multipliers, num_microbatches_options):
                if batch_size % num_microbatches != 0:
                    continue

                model = models[model_key]().to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr)

                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(train_data, train_labels),
                    batch_size=batch_size,
                    shuffle=True
                )

                privacy_engine = PrivacyEngine()
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=l2_norm_clip
                )
                privacy_statement = "Privacidade diferencial ativada"

                test_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(test_data, test_labels),
                    batch_size=batch_size,
                    shuffle=False
                )

                start_time = time.time()
                train_model(model, optimizer, criterion, train_loader, epochs)
                end_time = time.time()

                acc, prec, rec, f1 = evaluate_model(model, test_loader)

                try:
                    epsilon = privacy_engine.get_epsilon(delta)
                    privacy_statement = f"Epsilon: {epsilon:.2f}"
                except Exception as e:
                    privacy_statement = f"Erro ao calcular epsilon: {e}"

                with open(file_path, "a") as f:
                    f.write("------------------------------------------------------------------\n")
                    f.write(f"Teste {test_number}: \n\n")
                    f.write(f"Modelo: {model_key}\n")
                    f.write(f"DP: {dp} | LR: {lr} | Batch: {batch_size} | "
                            f"L2 Clip: {l2_norm_clip} | Noise: {noise_multiplier} | Microbatches: {num_microbatches}\n")
                    f.write(f"Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")
                    f.write(f"Tempo de Treinamento: {end_time - start_time:.2f} seg\n")
                    f.write(f"Privacidade: {privacy_statement}\n\n")

                test_number += 1

        else:
             
            model = models[model_key]().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_data, train_labels),
                batch_size=batch_size,
                shuffle=True
            )

            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_data, test_labels),
                batch_size=batch_size,
                shuffle=False
            )

            start_time = time.time()
            train_model(model, optimizer, criterion, train_loader, epochs)
            end_time = time.time()

            acc, prec, rec, f1 = evaluate_model(model, test_loader)

            privacy_statement = "Sem privacidade diferencial"

            with open(file_path, "a") as f:
                f.write("------------------------------------------------------------------\n")
                f.write(f"Teste {test_number}: \n\n")
                f.write(f"Modelo: {model_key}\n")
                f.write(f"DP: {dp} | LR: {lr} | Batch: {batch_size} | "
                        f"L2 Clip: N/A | Noise: N/A | Microbatches: N/A\n")
                f.write(f"Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")
                f.write(f"Tempo de Treinamento: {end_time - start_time:.2f} seg\n")
                f.write(f"Privacidade: {privacy_statement}\n\n")

            test_number += 1

print(f"Testes concluídos! Resultados salvos em '{file_path}'.")
import argparse
import os

import torch
import torchtvision
from tqdm import tqdm

from executorch.exir import EdgeCompileConfig, to_edge
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.export.experimental import _export_forward_backward

class CIFAR10Model(torch.nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class ModuleWithLoss(torch.nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.loss_fn = criterion

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        output = self.model(input)
        loss = self.loss_fn(output, target)
        return loss, output.detach().argmax(dim=1)
    
def get_data_loaders(batchsize: int = 4, num_workers: int = 2, data_dir: str = "./data"):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transforms
    )

    indices = torch.randperm(len(trainset))
    val_set_size = int(len(trainset) * 0.1)
    train_set_indices, val_set_indices = (
        indices[val_set_size:].tolist(),
        indices[:val_set_size].tolist(),
    )
    train_loader = torchvision.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers
    )
    val_loader = torchvision.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers
    )
    test_loader = torchvision.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, epochs: int = 10, lr: float = 0.001, momentum: float = 0.9, save_path:str = "./model.pt"):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for data in trainloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
        avg_epoch_loss = epoch_loss / len(trainloader)
        epoch_acc = epoch_correct / epoch_total * 100
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # testing 
        if testloader is not None:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.detach().item()

                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(testloader)
            test_acc = test_correct / test_total * 100
            print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # save the model with best loss for testing
        if testloader is not None:
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved model with loss {best_loss:.4f}")

    return model

def export_model(net, input_tensor, label_tensor):
    criterion = torch.nn.CrossEntropyLoss()
    model_with_loss = ModuleWithLoss(net, criterion)
    exported_program = export(model_with_loss, (input_tensor, label_tensor))
    exported_program = _export_forward_backward(exported_program)
    edge_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(
           _check_ir_validity=False
        )
    )
    return edge_program.to_executorch()

def save_model(ep, model_path):
    with open(model_path, "wb") as f:
        f.write(ep.buffer)
    
def load_model(model_path):
    with open(model_path, "rb") as f:
        model_bytes = f.read()
        et_mod = _load_for_executorch_from_buffer(model_bytes)
    return et_mod

def fine_tune_cifar10_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    return model

import torch.nn.functional as F
from tqdm import tqdm
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

def fine_tune_executorch_model(model_path, train_loader, val_loader, epochs=10, learning_rate=0.001):
    with open(model_path, "rb") as f:
        model_bytes = f.read()
        et_mod = _load_for_executorch_from_buffer(model_bytes)

    grad_start = et_mod.run_method("__et_training_gradients_index_forward", [])[0]
    param_start = et_mod.run_method("__et_training_parameters_index_forward", [])[0]

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        et_mod.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader):
            inputs, labels = batch

            # Forward pass
            out = et_mod.forward((inputs, labels), clone_outputs=False)
            loss = out[0]
            epoch_loss += loss.item()

            # Update parameters
            with torch.no_grad():
                for grad, param in zip(out[grad_start:param_start], out[param_start:]):
                    param.sub_(learning_rate * grad)

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Evaluate on validation set
        et_mod.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                out = et_mod.forward((inputs, labels))
                loss = out[0]
                val_loss += loss.item()
                predictions = out[1].argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    return et_mod

def main():
    model = CIFAR10Model()
    train_loader, val_loader, test_loader = get_data_loaders()

    # model training
    model = train_model(model, train_loader, val_loader, epochs=10, lr=0.001, save_path="./cifar10_model.pt")
    # Load the model
    model_path = "./cifar10_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Model file {model_path} does not exist. Training a new model.")
        model = CIFAR10Model()
        model = train_model(model, train_loader, val_loader, epochs=10, lr=0.001, save_path=model_path)

    # Fine-tune the model
    model = fine_tune_cifar10_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001)

    # Export the model
    validation_sample_data = next(iter(val_loader))
    img, lbl = validation_sample_data
    sample_input = img[0:1, :]
    sample_label = lbl[0:1]
    ep = export_model(model, sample_input, sample_label)
    save_model(ep, "./model.et")
    et_mod = load_model("./model.et")
    print(et_mod.run_method("forward", (sample_input, sample_label)))

    # Fine-tune the ExecuTorch model
    model_path = "./model.et"
    et_mod = fine_tune_executorch_model(model_path, train_loader, val_loader, epochs=10, learning_rate=0.001)

    # Test the fine-tuned model
    validation_sample_data = next(iter(val_loader))
    img, lbl = validation_sample_data
    sample_input = img[0:1, :]
    sample_label = lbl[0:1]
    print(et_mod.run_method("forward", (sample_input, sample_label)))

if __name__ == "__main__":
    main()

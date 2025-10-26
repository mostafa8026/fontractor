import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def make_synthetic_data(n_samples: int = 100, noise_std: float = 0.5):
    torch.manual_seed(0)
    # True underlying parameters: y = slope * x + intercept
    slope = 3.0
    intercept = 2.0

    x = torch.linspace(-5, 5, n_samples).unsqueeze(1)  # shape (n_samples, 1)
    noise = torch.randn_like(x) * noise_std
    y = slope * x + intercept + noise
    return x, y, slope, intercept


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train(model, dataloader, loss_fn, optimizer, epochs: int = 300):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in dataloader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataloader.dataset)
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.6f}")


def main():
    x, y, true_slope, true_intercept = make_synthetic_data(n_samples=200, noise_std=0.8)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegressionModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print("Initial parameters:", {name: p.data.numpy().ravel().tolist() for name, p in model.named_parameters()})
    train(model, dataloader, loss_fn, optimizer, epochs=5000)
    print("Trained parameters:", {name: p.data.numpy().ravel().tolist() for name, p in model.named_parameters()})
    print(f"True slope: {true_slope}, True intercept: {true_intercept}")

    # Example prediction
    model.eval()
    with torch.no_grad():
        x_new = torch.tensor([[-4.0], [0.0], [4.0]])
        preds = model(x_new)
        print("Inputs:", x_new.squeeze().tolist())
        print("Predictions:", preds.squeeze().tolist())

    # Save model state
    torch.save(model.state_dict(), "linear.pth")
    print("Model saved to linear.pth")


if __name__ == "__main__":
    main()
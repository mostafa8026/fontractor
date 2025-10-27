import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


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
    train(model, dataloader, loss_fn, optimizer, epochs=500)
    print("Trained parameters:", {name: p.data.numpy().ravel().tolist() for name, p in model.named_parameters()})
    print(f"True slope: {true_slope}, True intercept: {true_intercept}")

    # Plot data and fitted line
    model.eval()
    with torch.no_grad():
        # Create a smooth x range for the fitted line
        x_min, x_max = x.min().item(), x.max().item()
        x_plot = torch.linspace(x_min, x_max, 200).unsqueeze(1)
        y_plot = model(x_plot)

        # Convert to numpy for matplotlib
        x_np = x.numpy().ravel()
        y_np = y.numpy().ravel()
        x_plot_np = x_plot.numpy().ravel()
        y_plot_np = y_plot.numpy().ravel()

        plt.figure(figsize=(8, 6))
        plt.scatter(x_np, y_np, alpha=0.6, label="Data")
        plt.plot(x_plot_np, y_plot_np, color="red", linewidth=2, label="Fitted line")
        # Optionally plot true underlying line
        y_true_np = true_slope * x_plot_np + true_intercept
        plt.plot(x_plot_np, y_true_np, color="green", linestyle="--", linewidth=1.5, label="True line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Data and learned linear model")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Example prediction
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
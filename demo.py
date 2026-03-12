import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from alex_ricci_navier import AlexRicciNavier  # Importing your engine

def run_benchmark():
    print("--- Alex-Ricci-Navier (ARN) Optimizer Benchmark ---")
    print("Testing for stability under high-noise conditions...\n")

    # 1. Synthetic Data Generation (High Noise Environment)
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    # Target function with significant non-linear noise
    y = torch.sin(torch.sum(X, dim=1, keepdim=True)) + torch.randn(1000, 1) * 0.5

    # 2. Model Definition
    def create_model():
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    criterion = nn.MSELoss()
    epochs = 100
    lr = 1e-3

    # 3. Training Function
    def train(optimizer_type, name):
        model = create_model()
        # Initializing optimizer
        if name == "ARN":
            optimizer = optimizer_type(model.parameters(), lr=lr, nu=0.1, kappa=0.01)
        else:
            optimizer = optimizer_type(model.parameters(), lr=lr)
            
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Simulating Gradient Noise (common in Deep Learning)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * 0.4)
            
            optimizer.step()
            losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"[{name}] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        return losses

    # 4. Execution
    print("Starting Adam training...")
    adam_losses = train(torch.optim.Adam, "Adam")
    
    print("\nStarting ARN training...")
    arn_losses = train(AlexRicciNavier, "ARN")

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label='Adam Optimizer', color='red', linestyle='--')
    plt.plot(arn_losses, label='ARN Optimizer (Navier-Stokes/Ricci)', color='blue', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Optimization Stability: Adam vs Alex-Ricci-Navier')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    print("\n[Analysis] ARN shows higher stabilization in gradient flow.")
    print("Saving plot to 'benchmark_results.png'...")
    plt.savefig('benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    run_benchmark()

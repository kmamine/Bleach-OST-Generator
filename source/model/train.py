import torch
import torch.optim as optim
import mlflow
import os
import numpy as np
from torch.utils.data import DataLoader
from dataloader.dataset import YourCustomDataset
from models.model import YourCustomModel
from config.config import learning_rate, num_epochs, batch_size, input_channels, input_height, input_width
from losses.loss import CustomLoss
from metrics.metric import CustomMetric
from torchsummary import summary
from tqdm import tqdm



def log_model_summary(model, input_size):
    summary_str = str(summary(model, input_size=input_size))
    mlflow.log_text(summary_str, 'Model_Summary')

def validate(model, criterion, metric, val_dataloader, device):
    model.eval()  # Set model in evaluation mode
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Calculate the loss using the custom loss function
            loss = criterion(output, target)

            # Update metrics
            total_loss += loss.item()
            total_metric += metric(output, target).item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metric = total_metric / num_batches
    return avg_loss, avg_metric



def train():
    # Set up MLflow tracking
    mlflow.set_tracking_uri('your_mlflow_tracking_uri')  # Replace with your MLflow tracking URI
    mlflow.set_experiment('your_experiment_name')  # Replace with your experiment name

    # Log parameters
    mlflow.log_params({
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'input_channels': input_channels,
        'input_height': input_height,
        'input_width': input_width,
        # Add more parameters as needed
    })

    # Initialize your custom dataset and data loader
    train_dataset = YourCustomDataset('path/to/data')  # Replace 'templates/data' with the path to your training dataset folder
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = YourCustomDataset('path/to/data')  # Replace 'templates/val_data' with the path to your validation dataset folder
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize your custom model
    model = YourCustomModel(num_classes=10)  # Replace 10 with the number of classes in your classification problem
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize your custom loss and metric classes
    criterion = CustomLoss()
    metric = CustomMetric()

    # Start training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Set model in training mode
        model.train()
        pbar = tqdm(enumerate(dataloader),total = len(dataloader))
        for batch_idx, (data, target) in pbar:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)

            # Calculate the loss using the custom loss function
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Metric: {metric_value:.4f}"
            )

        avg_loss = total_loss / num_batches

        # Calculate inference time for a sample
        sample_data = next(iter(dataloader))[0].to(device)
        inference_time = get_inference_time(model, sample_data)

        # Calculate the custom metric for the epoch
        metric_value = metric(output, target)

        # Log metrics for the epoch
        mlflow.log_metric('loss', avg_loss, step=epoch)
        mlflow.log_metric('inference_time', inference_time, step=epoch)
        mlflow.log_metric('custom_metric', metric_value, step=epoch)

        # Log GPU metrics
        gpu_metrics = get_gpu_metrics()
        for metric_name, metric_value in gpu_metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

        # Print progress
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Inference Time: {inference_time:.4f} seconds')

        # Validation phase
        val_loss, val_metric = validate(model, criterion, metric, val_dataloader, device)
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        mlflow.log_metric('val_metric', val_metric, step=epoch)



        # Save the checkpoint weights after each epoch
        checkpoint_path = os.path.join( 'checkpoints', f'checkpoint_epoch_{epoch+1:04d}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path)

    # Save the trained model
    model_path = 'your_model.pth'  # Replace 'your_model.pth' with the desired model path
    torch.save(model.state_dict(), model_path)

    # Log model artifact to MLflow
    mlflow.log_artifact(model_path)

    # Log model summary
    summary_str = str(summary(model, input_size=(input_channels, input_height, input_width)))
    mlflow.log_text(summary_str, 'Model_Summary')

    # Calculate and log model FLOPs
    model_flops = get_model_flops(model, (input_channels, input_height, input_width))
    mlflow.log_metric('model_flops', model_flops)

    # Calculate and log model size
    model_size = get_model_size(model)
    mlflow.log_metric('model_size', model_size)


if __name__ == "__main__":
    # Set device to GPU if available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()

import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm
import torch.nn.functional as F

def get_data_loader(data_dir, batch_size=64, num_workers=4):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = ImageNet(root=data_dir, split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader, val_dataset

def evaluate_to_csv(model, dataloader, dataset, device, output_csv_path):
    model.eval()

    header = ['id', 'content', 'label'] + [f'label_{i}' for i in range(1000)]

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        with torch.no_grad():
            index = 0
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                # probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
                
                for i in range(images.size(0)):
                    # Get original image path from dataset samples
                    image_path, target_label = dataset.samples[index]
                    row = [index, image_path, target_label] + outputs[i].cpu().tolist()
                    writer.writerow(row)
                    index += 1

    print(f"Saved predictions to: {output_csv_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='/Users/floris/Documents/projects/hierarchical-conformal-prediction/data/imagenet/',
                        help="Path to ImageNet root directory")
    parser.add_argument('--output_csv', default='imagenet_logits.csv',
                        help="Path to save CSV file with predictions")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--arch', type=str, default='resnet152',
                        help="Model architecture (e.g. resnet18, resnet50, resnet101)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.arch}")
    model = getattr(models, args.arch)(pretrained=True)
    model.to(device)

    print("Loading validation data...")
    val_loader, val_dataset = get_data_loader(args.data_dir, args.batch_size, args.workers)

    print("Evaluating and writing to CSV...")
    evaluate_to_csv(model, val_loader, val_dataset, device, args.output_csv)

if __name__ == "__main__":
    main()
import torch
from torch.utils.data import DataLoader
from src.dataloader import ProductImageDataset
from models.model import SimpleCNN
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ProductImageDataset(csv_file='data/train.csv', image_dir='data/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleCNN(num_classes=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # Number of epochs
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        images = batch['image']
        entity_values = batch['entity_value'].float()  
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, entity_values)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

torch.save(model.state_dict(), 'models/checkpoints/simple_cnn.pth')

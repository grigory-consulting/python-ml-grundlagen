from torchvision import datasets, transforms 

transform = transforms.Compose([
    transforms.ToTensor(), # Konvertiert Bilder zu Tensoren
    transforms.Normalize((0.5,), (0.5, ))
 ] # Normalierung der Daten  
)

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform) # Trainingsdaten
test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform) # Testdaten 

train_loader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle=True) # shuffle = Randomisieren, 
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle=False) #


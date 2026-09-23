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



import matplotlib.pyplot as plt
klassen = ["T-Shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
           "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefelette"]
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for nummer, ax in enumerate(axes.flat):
    index = int((train_data.targets == nummer).nonzero()[0])
    ax.imshow(train_data[index][0][0], cmap="gray")
    ax.set_title(f"{nummer}: {klassen[nummer]}")
    ax.axis("off")
plt.show()

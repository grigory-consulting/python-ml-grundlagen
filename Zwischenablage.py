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



# Neuronales Netz mit zwei Linearschichten 

class NN(nn.Module): # 
    def __init__(self):
        super().__init__() # aus nn.Module erben 
        self.flatten = nn.Flatten() # 1x28x28 Pixel -> Vektor der Länge 784  
        self.layer1 = nn.Linear(28*28, 128) # 128 Neuronen (immer Zweierpotenzen)
        self.output = nn.Linear(128, 10) # 128 -> 10 Klassen 

    def forward(self,x): # wie werden die Daten vorwärts propagiert? 
        x = self.flatten(x)
        x = self.layer1(x) # x = Wx + b 
        x = torch.relu(x) # ReLU(x) 
        return self.output(x) # ohne Aktivierung, weil Multiklassen-Problem und wir nn.CrossEntropy verwenden

class NN(nn.Module): # 
    def __init__(self):
        super().__init__() # 
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self,x):
        return self.model(x)

def train_model(model, optimizer, n_epochs=10):
    loss_fn = nn.CrossEntropyLoss() # weil mehrere Klassen 
    verluste = [] 
    for epoch in range(n_epochs): # Training Loop... 1 Epoche = 1 Mal hat ML Algorithmus den Datensatz vollständig gesehen 
        model.train() # Trainingsmodus 
        running_loss = 0.0 # laufende Kosten 
        for images, labels in train_loader: # batchweise die Daten rausholen 
            optimizer.zero_grad()           # alte Gradienten löschen 
            outputs = model(images) # Vorwärtslauf
            loss = loss_fn(outputs, labels) # Verlust, wie schlecht ist mein Modell? 
            loss.backward() # Rückwärtslauf = Gradienten berechnen 
            optimizer.step() # Anpassung der Gewichte 
            running_loss += loss.item()
        verluste.append(running_loss / len(train_loader)) # Durchschnitt bilden 
        print(f"Epoche {epoch+1}/{n_epochs}, Verlust: {verluste[-1]:.4f}")

    return verluste

model = NN()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
verluste = train_model(model, optimizer, n_epochs=20)


# Confusion-Matrix 
import pandas as pd
wahr_name = pd.Series(true.numpy(), name="wahr").map(lambda i: klassen[i])
pred_name = pd.Series(pred.numpy(), name="vorhergesagt").map(lambda i: klassen[i])
tabelle = pd.crosstab(wahr_name, pred_name).reindex(index=klassen, columns=klassen, fill_value=0)
tabelle

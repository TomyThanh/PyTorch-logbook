from ptFUNCTIONS import *


"""EXERCISE / SECOND EXAMPLE"""
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup
weight = 0.7
bias = 0.3

X = torch.arange(0, 10, 0.02)
y = weight * X + bias

X_train = X[:400].reshape(-1, 1)  # reshape für (batch_size, input_features)
X_test = X[400:].reshape(-1, 1)  # reshape für (batch_size, input_features)
y_train = y[:400].reshape(-1, 1)  # reshape, damit es der Ausgabe entspricht
y_test = y[400:].reshape(-1, 1)  # reshape für die Testdaten

#1. Building the Model
class LinearRegressionModelV2(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linearLayer = nn.Linear(in_features=1, out_features=1)
      
    def forward(self, x):
        return self.linearLayer(x)

model1 = LinearRegressionModelV2()

#2. Compiling
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.001)

#3. Fitting our model
epochs = 5000
epoch_count, loss_values, testLossValues, test_pred, testLoss = train_and_test_loop(model1, X_train, y_train, X_test, y_test, 
                                                                                    optimizer, loss_fn, epochs)

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=test_pred)
print()
print()

# Beispiel für einen Custom-Wert, den du vorhersagen möchtest
custom_value = torch.tensor([[5.0]], dtype=torch.float32)  # Beispiel für x = 5 (bei LinearRegression)

model1.eval()  
with torch.inference_mode():  
    prediction = model1(custom_value)  
    print(f"Vorhersage für x = {custom_value.item()}: {prediction.item()}")

#Saving our model
torch.save(model1.state_dict(),"c:/Unkram123/pytorch_env/pth_models/model1.pth" )

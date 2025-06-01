from ptFUNCTIONS import *


"""ＰＹＴＯＲＣＨ ＷＯＲＫＦＬＯＷ"""

#Prepare data / splitting to Training-, Validation- and Test-sets --> preprocessing data
#Building model
#Fitting the model to data
#Evaluate the model (visualizing)
#Making predictions 

from torch import nn #nn contains all building-models to build a neural network

#1. Data (preparing and loading)
#We'll use the linear regression formula to make a straight line

weight = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

#2. Splitting data into training and test-sets

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) RANDOM!!!!
X_train = X[:40]
y_train = y[:40]
X_test = X[40:]
y_test = y[40:]

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test)

#3. FIRST PYTORCH MODEL
""" BUILDING OUR MODEL """
class LinearRegressionModel(nn.Module): #<-- almost everything in PyTorch inherit from nn.Module

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias #This is the linear regression formula which it should learn

#PyTorch model building essentials
# torch.nn - contains all of the buildings blocks of neural networks
# torch.nn.Parameter - what parameters should our model try and learn
# torch.nn.Module - The base class for all neural network modules, please overwrite your forward()-method
# torch.optimm - the optimizer of PyTorch
# def forward() - All nn.Module subclasses require you to overwrite forward(), defines what happens in the forward computation


#Checking out the contents of our PyTorch model by creating the first OBJECT of the MODEL CLASS
torch.manual_seed(42)
model0 = LinearRegressionModel()
list(model0.parameters()) #shows us the parameters
model0.state_dict() #Shows us the parameters

#Making prediction using 'torch.inference_mode()'
#When we pass data thorugh our model, it's going to run it through the forward()-method
with torch.inference_mode():
    y_preds = model0(X_test)

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)


"""--- COMPILING OUR MODEL ---"""
#Set up a loss function
loss_fn = nn.L1Loss()

#Set up an optimizer
optimizer = torch.optim.Adam(params=model0.parameters(), 
                                lr=0.01)


"""-- FITTING OUR MODEL --"""
"""TRAINING"""
#0. Loop through the data
#1. Forward pass (data moving through our forward()-function)
#2. Calculate the loss 
#3. Optimizer zero grad
#4. Loss backward - move backwards through the network to calculate the gradients of each parameters (**backpropagation**)
#5. Optimizer step - improves the loss (gradient descent)

epochs = 2000
epoch_count = []
loss_values = []
testLossValues = []
for epoch in tqdm(range(epochs), desc="Training läuft"):
    
    #0. Put the model in training 
    model0.train()

    #1. Forward pass on train data using forward()
    y_pred = model0(X_train)

    #2. Calculate the loss 
    loss = loss_fn(y_pred, y_train)

    #3. Zero gradients of the optimizer 
    optimizer.zero_grad()

    #4. Perfrom backpropagation on the loss
    loss.backward()

    #5. Progress / step the optimiezer
    optimizer.step()
    print(f"Epochs {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


    """TESTING"""
    model0.eval()                            #turns off different settings in model not needed for evaluation
    with torch.inference_mode():             #PREDICTING OUR RESULTS
            
            #1. Do the forward pass / RESULTS OF PREDICTION
            test_pred = model0(X_test)

            #2. Calculate the loss / RESULTS
            testLoss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:                      #Visualizing loss and epochs
        epoch_count.append(epoch)
        loss_values.append(loss)
        testLossValues.append(testLoss)

        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {testLoss}")

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=test_pred)

#Plot the loss curves
plt.plot(epoch_count, torch.tensor(loss_values).numpy(), label="Train loss")
plt.plot(epoch_count, torch.tensor(testLossValues).numpy(), label = "Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()


"""SAVING A MODEL IN PYTORCH"""
#1. 'troch.save()' - allows you to save a PyTorch OBJECT in Python's pickle format
#2. 'torch.load()' - allows you to laod a saved PyTorch object
#3. 'torch.nn.Module.load_state_dict()' - this allows to load a model's saved state dictionary

torch.save(model0.state_dict(),"c:/Unkram123/pytorch_env/pth_models/model0" )

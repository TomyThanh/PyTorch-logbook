from ptFUNCTIONS import *
from sklearn.datasets import make_circles

"""NEURAL NETWORK BINARY CLASSIFICATION WITH PYTORCH"""
#Binary Classification - two classes (e.g cat vs.dog, spam vs not spam, ...)
#Multi class classification - more than two classes

#0. Data
nSamples = 1000
X, y = make_circles(nSamples, noise=0.03, random_state=42)

circles = pd.DataFrame({"X1": X[:, 0], 
                        "X2": X[:,1],
                        "label": y})
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()

#Turn data into tensors and train-test-split
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = torch.reshape(y_train, (-1, 1))
y_test = torch.reshape(y_test, (-1, 1))

device = "cuda" if torch.cuda.is_available() else "cpu"

#1. Building the Model
class CircleModelV0(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.linear(x) 

model0 = CircleModelV0()

#2. Compiling Loss and Optimizer
loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogits automatically applies sigmoid inside the loss function
optimizer = torch.optim.Adam(params=model0.parameters(), lr=0.01)
def accuracy_fn(y_true, y_pred):
    # Round the predictions to binary values and compare with the true labels
    y_pred = torch.round(torch.sigmoid(y_pred))  # apply sigmoid before rounding
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

#3. Training the Model
epoch_count, loss_values, testLossValues, test_pred, testLoss = train_and_test_loop(model0, X_train, y_train, X_test, y_test,
                                                                                    loss_fn, optimizer, 10)

plot_decision_boundary(model=model0, X=X_test, y=y_test)

###-- IMPROVEMENTS ---#
#Change the loss function
#Change the optimization function
#Change the learning rate
#Change the activation function
#Adding more Neurons
#Adding more Hidden Layers
#Fitting for longe (more epochs)

"""NON-LINEARITY BINARY CLASSIFICATION NEURAL NETWORK"""
#BASICALLY JUST USING RELU-FUNCTION()
#1. Building
class CircleModelV1(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32,1)
        )
    
    def forward(self, x):
        return self.layer(x)

model1 = CircleModelV1()

#2. Compiling
loss_fn2 = nn.BCEWithLogitsLoss()
optimizer2 = torch.optim.Adam(params=model1.parameters(), lr=0.1)

#3. Fitting
epoch_count, loss_values, testLossValues, test_pred, testLoss = train_and_test_loop(model1, X_train, y_train, X_test, y_test,
                                                                                    loss_fn2, optimizer2, 500)


#=== REPLICATING NON-LINEAR ACTIVATION FUNCTIONS ===#
A = torch.arange(-10, 10, 1).type(torch.float32)
plt.plot(A) #Linear

def relu(x): #relu-maker
    return torch.max(torch.tensor(0), x)
plt.plot(relu(A))

def sigmoid(x): #sigmoid-maker
    return 1 / (1 + torch.exp(-x))
plt.plot(sigmoid(A))



from ptFUNCTIONS import *


"""MULTICLASS CLASSIFICATION WITH PYTORCH"""

#0. Data
from sklearn.datasets import make_blobs
NUM_CLASSES = 4
NUM_FEATURES = 2

X, y = make_blobs(n_samples=1000, 
                  n_features=NUM_FEATURES,
                  centers=NUM_CLASSES,
                  cluster_std=1.5,
                  random_state=42)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(10,7))
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.title("Train data")

#1. Building
class BlobModel(nn.Module):

    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear(x)

model4 = BlobModel(input_features=2, output_features=4)

#2. Compiling
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model4.parameters(), lr=0.001)

#3. Fitting
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

epochs = 1500
for epoch in range(epochs):
    model4.train()
    y_preds = model4(X_train)
    loss = loss_fn(y_preds, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=torch.softmax(y_preds, dim=1).argmax(dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    model4.eval()
    with torch.inference_mode():
        test_preds = model4(X_test)
        testLoss = loss_fn(test_preds, y_test)
        testAcc = accuracy_fn(y_true=y_test, y_pred=torch.softmax(test_preds, dim=1).argmax(dim=1))
    
    test_losses.append(testLoss.item())
    test_accuracies.append(testAcc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item()} , Test Loss: {testLoss.item()} | Acc: {acc}% , Test Acc: {testAcc}%")
        
#4. Getting the prediction probabilities for a multi-class model
# Logits (raw output) -> Pred probs (use 'torch.softmax') -> Pred labels (use 'torch.argmax()')

#Plotting the learning curve


plot_learning_curve(train_losses=train_losses, test_losses=test_losses, train_accuracies=train_accuracies, test_accuracies=test_accuracies, epochs=epochs)


#MAKING PREDICTIONS
model4.eval()
plot_decision_boundary(model4, X_test, y_test)



"""A FEW MORE CLASSIFICATION METRICS IN PyTorch"""











































































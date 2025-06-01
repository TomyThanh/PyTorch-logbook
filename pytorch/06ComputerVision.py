from ptFUNCTIONS import *



"""PYTORCH COMPUTER VISION MODELV1 (NON-LINEARITY)"""

#0. Getting a dataset (FashinMNIST from torchvision.datasets) AND preprocessing
train_data = datasets.FashionMNIST(
    root="data" , #where to download data to?
    train=True,
    download=True,
    transform=transforms.ToTensor(), #How do we want to transform the data?
    target_transform=None #How do we want to transfrom the labels?
)

test_data = datasets.FashionMNIST(
    root="data" , #where to download data to?
    train=False,
    download=True,
    transform=transforms.ToTensor(), #How do we want to transform the data?
    target_transform=None #How do we want to transfrom the labels?
)

train_dataLoader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = class_names = train_data.classes


#1. BUILDING A BETTER MODEL WITH LINEARITY AND NON-LINEARITY
class FashionMNISTV1(nn.Module):
    
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #flattens input into a single vector
            nn.Linear(in_features=input_shape, out_features=128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32, out_features=output_shape)
            )

    def forward(self, x):
        return self.layer_stack(x)

model1 = FashionMNISTV1(input_shape=28*28, output_shape=len(class_names))


#2. + 3. COMPILING AND FITTING
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.001)

"""
epochs = 3
for epoch in tqdm(range(epochs), desc="Training läuft"):
    trainLoss = 0
    print(f"Epoch: {epochs}\n-----")

    for batch, (x_train, y_train) in enumerate(train_dataLoader):
        
        model1.train()
        y_pred = model1(x_train)
        loss = loss_fn(y_pred, y_train)
        trainLoss = trainLoss + loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 ==0:
            print(f"Looked at {batch * len(x_train)} / {len(train_data)} samples.")

    trainLoss = trainLoss / len(train_dataLoader)

    
    #===================================TESTING LOOP===================================#
    model1.eval()
    with torch.inference_mode():
        testLoss, testAcc = 0,0
        for batch, (x_test, y_test) in enumerate(test_dataLoader):

            y_pred_test = model1(x_test)
            loss = loss_fn(y_pred_test, y_test)
            acc = accuracy_fn(y_pred_test, y_test)
            testLoss = testLoss + loss
            testAcc = testAcc + acc
        
        testLoss = testLoss / len(test_dataLoader)
        testAcc = testAcc / len(test_dataLoader)
    print(f"\nTrain loss: {trainLoss:.4f} | Test Loss: {testLoss:.4f}, Test acc: {testAcc:.0f}%")
"""  



"""TURNING OUR TRAINING LOOP AND TESTING LOOP INTO FUNCTIONS"""
#Training loop - 'train_step()'
#Testing loop - 'test_step()'

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: accuracy_fn,
               device: torch.device):
    """Performs a training with model trying to learn on data_loader"""
    train_loss, train_acc = 0,0
    model.train()
    for batch, (x_train, y_train) in enumerate(data_loader):

        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        train_loss = train_loss + loss
        train_acc = train_acc + accuracy_fn(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.0f}%")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn: accuracy_fn,
              device: torch.device):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for x_test, y_test in data_loader:
            
            y_pred_test = model(x_test)
            loss = loss_fn(y_pred_test, y_test)
            acc = accuracy_fn(y_pred_test, y_test)
            
            test_loss = test_loss + loss
            test_acc = test_acc + acc
        
        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test Acc: {test_acc:.0f}%")

#Trying our new functions
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_time_start = default_timer()
epochs = 3
for epoch in tqdm(range(epochs), desc="Training läuft"):
    print(f"Epoch: {epoch}\n------")
    train_step(model1, train_dataLoader, loss_fn, optimizer, accuracy_fn, device)
    test_step(model1, test_dataLoader, loss_fn, accuracy_fn, device)

train_time_end = default_timer()
total_train_time = print_train_time(train_time_start, train_time_end, device)
model1_results = eval_model(model1, test_dataLoader, loss_fn, accuracy_fn, device)
print(model1_results)


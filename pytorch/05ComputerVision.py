from ptFUNCTIONS import *
import torch.optim


"""PYTORCH COMPUTER VISION WITH TORCHVISION"""

#0. Computer vision libarries in PyTorch
# torchvision.datasets - get datasets and data loading functions for computer vision here
# torchvision.models - get pretrained computer vision models that you can use for your own problems
# torchvision.transforms - functions for manipulating your vision data (images) to be suitable to use with an ML Model
# torch.utils.data.Dataset - Base dataset class for pytorch 
# torch.utils.data.DataLoader - Creates a Python iterable over a dataset


#1. Getting a dataset (FashinMNIST from torchvision.datasets)
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

#Check the shape of our image
image0, label0 = train_data[0]
class_names = train_data.classes
class_names
image0.shape

#1.2 Visualizing our data
plt.imshow(image0.squeeze(), cmap="gray")
plt.title(class_names[label0])
plt.axis(False)


#Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    randomIDX = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[randomIDX]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis("off")



#2. Prepare DataLoader
# Right now, our data is in the form of PyTorch Datasets
# DataLoader turns our dataset into a Python iterable
# We want to turn our data into batches (mini-batches)
# We break down our 60000 images into 32 images per portion

train_dataLoader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=32, shuffle=False)

#Check out what's inside the training dataLoader
train_features_batch, train_labels_batch = next(iter(train_dataLoader))
train_features_batch.shape, train_labels_batch.shape

#Show a random sample
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("off")



"""BUILDING OUR FIRST MODEL (BASIC MODEL)"""

#Additional example
#Create a flatten layer
flatten_model = nn.Flatten()
x = train_features_batch[0]
flatten_model(x) #--> [color_channel, height*width]

#1. BUILDING
class FashionMNISTModelV0(nn.Module):

    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=32),

            nn.Linear(32, out_features=output_shape)
    )

    def forward(self, x):
        return self.layer_stack(x)

model0 = FashionMNISTModelV0(input_shape=784, output_shape=len(class_names)) #one for every class
dummy = torch.rand(size=[1,1,28,28])


#2. + 3. COMPILING & FITTING
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model0.parameters(), lr=0.001)

#We're going to calculate the train loss ***per batch**
from tqdm.auto import tqdm
start_timer = default_timer()
epochs = 3

#Epoch Loop
for epoch in tqdm(range(epochs), desc="Training läuft"):
    print(f"Epoch: {epochs}\n-----")

    #TRAINING per BATCH / BATCH LOOP
    train_loss = 0
    #Add a loop to loop through the training batches
    for batch, (x_train, y_train) in enumerate(train_dataLoader): #BATCH LOOPING
        model0.train()

        y_pred = model0(x_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss #stacks train loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(x_train)} / {len(train_data)} samples.")

    #Divide total train loss by length of train dataLoader
    train_loss /= len(train_dataLoader)

    #==========================TESTING LOOP==========================#
    test_loss, test_acc = 0,0
    model0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataLoader: #BATCH LOOP
            test_pred = model0(X_test)

            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(test_pred, y_test)

        #Calculate the test loss average per batch
        test_loss /= len(test_dataLoader)
        test_acc /= len(test_dataLoader)
    print(f"\nTrain loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

traintime_end = default_timer()
totalTraintimeV0 = print_train_time(start_timer, traintime_end)

"""4. MAKE PREDICTIONS AND GET MODEL 0 RESULTS"""
def eval_model(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn: accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred, y)

            #Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,  # only works when model was created with class
            "model_loss": loss.item(),
            "model_acc": round(acc)}

model0_results = eval_model(model0, test_dataLoader, loss_fn, accuracy_fn)
print(model0_results)        





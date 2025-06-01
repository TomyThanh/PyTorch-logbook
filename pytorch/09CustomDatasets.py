from ptFUNCTIONS import *



"""TRANSFORMS AND DATA AUGMENTATION"""
# Data augmentation is the process of artificially adding diversity to your training data.
# For our case it means applying various image transfromations
# In a nutshell the model is able to see the images in different kinds of perspectives


# 0. PREPROCESSING DATA
train_transformer = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

test_transformer = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize( # Converts the values to numbers between 0 and 1 depending on the value
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_data_simple = ImageFolder(root="C:/Unkram123/food_data/pizza_steak_sushi/train", transform=train_transformer)
test_data_simple = ImageFolder(root="C:/Unkram123/food_data/pizza_steak_sushi/test", transform=test_transformer)

train_dataloader_simple = DataLoader(train_data_simple, batch_size=32, shuffle=True)
test_dataloader_simple = DataLoader(test_data_simple, batch_size=32, shuffle=False)

class_names = train_data_simple.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. BUILDING
class TinyVGG(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(),

            nn.Conv2d(hidden_units, hidden_units,
                      kernel_size=3, padding=0, stride=1),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(),

            nn.Conv2d(hidden_units, hidden_units,
                      kernel_size=3, padding=0, stride=1),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5408 , out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

model_0 = TinyVGG(input_shape=3, output_shape=len(class_names), hidden_units=32)
model_0 = model_0.to(device)


### Try Forward on a single image batch
image_batch, label_batch = next(iter(train_dataloader_simple))
model_0(image_batch)



### CREATING TRAINING AND TESTING LOOP FUNCTIONS ###
def train_function(model: torch.nn.Module,
                   train_dataloader: torch.utils.data.DataLoader,
                   test_dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn.Module,
                   accuracy_fn: accuracy_fn,
                   epochs: int,
                   device=device):
    
    # 2. Create an empty results dictionary
    results = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs), desc="Training läuft"):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, accuracy_fn, device)

        print(f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.0f}%")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results


# 2. + 3. COMPILING AND FITTING
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.0005)

train_time_start = default_timer()
epochs = 2

model_0_results = train_function(model_0, train_dataloader_simple,
                                 test_dataloader_simple, optimizer, loss_fn,
                                 accuracy_fn, epochs, device)

train_time_end = default_timer()
print(f"Total training time: {train_time_end-train_time_start:.2f}s")


# 4. EVALUATION
print(eval_model(model_0, test_dataloader_simple, loss_fn, accuracy_fn, device))

# Loss curve
def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

plot_loss_curves(model_0_results)


# Confusion Matrix
y_true = []
y_preds = []
model_0.eval()
with torch.inference_mode():
    for inputs, labels in test_dataloader_simple:

        outputs = model_0(inputs)
        _, preds = torch.max(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_preds.extend(preds.numpy())

cm = confusion_matrix(y_true, y_preds, labels=[0,1,2])
sns.heatmap(cm, annot=True, cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()



"""OVERFITTING AND UNDERFITTING"""
# With loss curves ('learning curves') we can see if our model is underfitting or overfitting
# OVERFITTING
# 1. More data or better data
# 2. Data augmentation
# 3. Transfer learning
# 4. Simplifying model
# 5. learning rate decay (slowly decreasing the learning rate: high learning rate --> low learning rate)
# 6. Early stopping (when the loss goes up for a while the model stops training)


# UNDERFITTING
# 1. More layers / units
# 2. Change the learning rate
# 3. Train for longer / more epochs
# 4. transfer learning
# 5. Balance between overfittign and underfitting


# COMPARING
# 1. PyTorch + Tensorboard
# 2. Weights + Biases API
# 3. MLFlow



"""PREDICTION ON CUSTOM DATA"""
# Predicting an image that is not in train data neither test data
# We have to turn to image to a tensor with read_image() and then transform as wel as dtype it so matrix multiplication is compatible

custom_image_transformer = transforms.Compose([
    transforms.Resize(size=(64,64))
])


def custom_predictions(class_names, model, paths: str, transformer: torchvision.transforms):

    """MAKES A PREDICTION ON YOUR OWN IMAGE"""
    custom_image = torchvision.io.read_image(paths)
    custom_image_transformers = transformer
    transformed_custom_image = custom_image_transformers(custom_image).type(torch.float32) / 255

    model.eval()
    with torch.inference_mode():
        custom_prediction = model(transformed_custom_image.unsqueeze(0))
    custom_pred_prob = torch.softmax(custom_prediction, dim=1)
    custom_pred_label = torch.argmax(custom_pred_prob, dim=1)

    plt.imshow(custom_image.permute(1,2,0).numpy())
    plt.title(f"Your Image is PROBABLY a {class_names[custom_pred_label]}")    
    plt.axis("off")
    plt.show()


custom_predictions(class_names=class_names, model=model_0,
                   paths="C:/Unkram123/food_data/pizza_steak_sushi/custom_predictions/italy-1264104_1920.jpg",
                   transformer=custom_image_transformer)






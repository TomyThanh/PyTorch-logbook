from ptFUNCTIONS import *



"""CONVOLUTIONAL NEURAL NETWORKS WITH PYTORCH"""

# CNN's are als known as ConvNets and are known for their capabilities to find patterns in visual data

#0. DATA PREPROCESSING
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

class_names = train_data.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. BUILDING
class FashionMNISTModelV2(nn.Module):
    """This model replicates TinyVGG"""

    def __init__(self, input_shape, output_shape):
        super().__init__()
        #Create a CONV-BLOCK
        self.conv_block_1 = nn.Sequential(
           
            nn.Conv2d(in_channels=input_shape, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64,
                      kernel_size=3,
                      stride = 1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            )
        
        self.conv_block_2 = nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32,16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)

model2 = FashionMNISTModelV2(1, output_shape=len(class_names))
model2 = model2.to(device)

#How to find out the INPUTSHAPE AND THE OUTPUTSHAPE
#rand_image_tensor = torch.randn(size=(1,64,64))
#model2(rand_image_tensor)

#2. + 3. COMPILING AND FITTING
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
train_time_start = default_timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    train_step(model2, train_dataLoader, loss_fn,
               optimizer, accuracy_fn, device=device)
    test_step(model2, test_dataLoader, loss_fn, accuracy_fn, device)

train_time_end = default_timer()
print(print_train_time(train_time_start, train_time_end, device))
model2_eval = eval_model(model2, test_dataLoader, loss_fn, accuracy_fn, device)
print(model2_eval)



"""COMPARING RESULTS AND MAKING PREDICTIONS"""
# compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
# compare_resutls["training time"] = [total_train_time_model_0, total_train_time_model_1, total_train_time_model_2]

# 9. Make and evaluate random predictions with best model

def make_predictions(model: nn.Module,
                     data: list,
                     device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #Prepare the sample
            sample = torch.unsqueeze(sample, dim=0).to(device)

            #Forward pass 
            pred_logit = model(sample)

            #Get prediction probability (logit -> pred prob)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            #Get pred off the GPU for further visualizations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_prohs to turn list into a tensor
    return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

#Make predictions
proof_predictions = make_predictions(model2, test_samples, device)

#Convert prediction propabilities into labels
pred_classes = proof_predictions.argmax(dim=1)

#Plot predictions
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # Find the prediction (in text form, e.g "Sandal")
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form)
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    titel_text = f"Pred: {pred_label} |  Truth: {truth_label}"

    # Check for equality between pred and turth and change color of title text
    if pred_label == truth_label:
        plt.title(titel_text, fontsize=10,c="g")
    else:
        plt.title(titel_text, fontsize=10,c="r")




"""======CONFUSION MATRIX==========CONFUSION MATRIX==========CONFUSION MATRIX==========CONFUSION MATRIX======"""
# Confusion matrix is really good at evaluating your classification models

# 1. Make predictions with trained model
y_preds = []
model2.eval()
with torch.inference_mode():
    for x_test, y_test in tqdm(test_dataLoader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        y_logit = model2(x_test)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

        y_preds.append(y_pred.cpu())

y_pred_tensor = torch.cat(y_preds)


# 2. Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor,
                         target = test_data.targets)


# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10,7))


#SAVING AND LOADING OUR MODEL
MODEL_PATH = Path("C:\Unkram123\models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cnn_modelv2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


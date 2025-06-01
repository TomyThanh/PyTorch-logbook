from ptFUNCTIONS import *


"""ⒸⓊⓈⓉⓄⓂ ⒹⒶⓉⒶⓈⒺⓉⓈ ⓌⒾⓉⒽ ⓅⓎⓉⓄⓇⒸⒽ"""
# How do we get our own data in PyTorch?
# One way to do is via: custom datasets
# Domain libaries: torchvision, torchaudio, torchtext, torchrec, etc. have DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting data
import requests
import zipfile
from pathlib import Path

# Setup path do a data folder
data_path = Path("food_data")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it....
if image_path.is_dir():
    print(f"{image_path} directory already exists.. skipping download")
else:
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download steak_sushi_pizza data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak and sushi data...")
    zip_ref.extractall(image_path)


# 2. Becoming one with the data / Visualizing our data
from PIL import Image

image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem
img = Image.open(random_image_path)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title(f"{image_class}")


# 0. PREPROCESSING DATA
# We have to turn our data into tensors
# We have to turn it into a 'torch.utils.data.Dataset' and subsequently a 'torch.utils.data.DataLoader'
# 'transforms.Compose' is like PHOTOSHOP

transformer = transforms.Compose([
    
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5), # Flips the images randomly horizontally
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    ])


def plot_transformed_images(image_paths, transform, n=3):
    """Selects random images from a path of images and loads/transforms them then plots the
    original vs the transformed one"""

    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis("off")

            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize= 16)
    plt.show()

#plot_transformed_images(image_path_list, transformer, )


# 4. USING OPTION 1: datasets.ImageFolder
#    USING OPTION 2: Building our own ImageFolder

train_data = datasets.ImageFolder(root="C:/Unkram123/food_data/pizza_steak_sushi/train", transform=transformer)
test_data = datasets.ImageFolder(root="C:/Unkram123/food_data/pizza_steak_sushi/test", transform=transformer)

# Get class names as a list
class_names = train_data.classes

## 4.1 Turn loaded images into DataLoader's
# A dataloader is goint to help us to turn  our Dataset into iterables.
train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)



"""CREATING OWN CUSTOM DATASET CLASS TO LOAD OUR IMAGES"""

# 1. Want to be able to load images from a file
# 2. Want to be able to get class names from the Dataset
# 3. Want to be able to get classes as dictionary from the Dataset

#PROS:
# Can create 'Dataset' out of almost anything
# Not limited to PyTorch pre-built 'Dataset' functions

#CONS:
# Even though you could create 'Dataset' of almost anything, it will not work 100%
# Using a custom 'Dataset' often results into writing more code, which could lead to more errors


## 5.1 Creating a hyperfunction to get class names
# 1. Get the class names using 'os.scandir()' to traverse a target directory
# 2. Raise an error if the class names aren't found 
# 3. Turn the class names into a dict and a list and return them

target_directory = "C:/Unkram123/food_data/pizza_steak_sushi/train"
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])


def find_classes(directory: str):
    """Finds the class folder names in a target directory."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Could not find any classes in {directory}... please checkt file structure")
    
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx


## 5.2 CREATE OWN ImageFolder
# 1. Subclass 'torch.utils.data.Dataset'
# 2. Init our subclass with a target directory
# 3. Create several attributes:
# paths - paths of our images
# transform - the transform we'd like to use
# classes - a list of the target classes
# class_to_idx - a dict of the target classes
# 4. Create a 'load_images()' function, this will open the image
# 5. Overwrite the '__len()__' to return the length of dataset
# 6. Overwrite the '__getitem()__' method to return a given sample when passed an index

# Own ImageFolder (customizable)
class ImageFolderCustom(Dataset):

    def __init__(self, targ_dir: str, transform=None):
        super().__init__() 
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transforms = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int):
        """Opens an image via a path and returns it."""
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        """Returns the total number of samples"""
        return len(self.paths)
    
    def __getitem__(self, index: int):
        """Returns one sample of data, data and label (X,y)"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

custom_transformer = transforms.Compose([

    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_data_custom = ImageFolderCustom(targ_dir="C:/Unkram123/food_data/pizza_steak_sushi/train", transform=custom_transformer)
test_data_custom = ImageFolderCustom(targ_dir="C:/Unkram123/food_data/pizza_steak_sushi/test", transform=custom_transformer)

train_data_custom.class_to_idx # Our own creation
train_data_custom.classes # Our own creation


# 5.3 Create a function to display random images
# 1. Take in a 'Dataset' and a number of other parameters 
# 2. To prevent the display getting out of hand, let's cap the number of images to 10
# 3. Get a list of random sample indexes from the target dataset.
# 4. Setup a matplotlib plot. 
# 5. Loop through the random sample images and plot them

def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool=True):
    
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes n shouldn't be larger than 10.")
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16,8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image_adjust = targ_image.permute(1,2,0)

        plt.subplot(1,n, i+1)
        plt.imshow(targ_image_adjust, cmap="gray")
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"

        plt.title(title)
    
    plt.show()

display_random_images(train_data_custom, class_names)

train_data_custom_loader = DataLoader(train_data_custom, batch_size=32, shuffle=True)
test_data_custom_loader = DataLoader(test_data_custom, batch_size=32, shuffle=False)

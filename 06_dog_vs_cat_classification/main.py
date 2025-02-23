from torchvision import transforms

from models import BasicModule
from data import DogVsCatDataset
from PIL import Image

root = './AllData/competitions/dog-vs-cat-classification'
dataset = DogVsCatDataset(root, mode='train')
print(len(dataset))
data, label = dataset[0]
print(data.shape)
print(label)
to_img = transforms.ToPILImage()
img = to_img(data)
img.show()

model = BasicModule()
path = model.save()
model.load(path)

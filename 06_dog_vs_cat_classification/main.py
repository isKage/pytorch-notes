from data import DogVsCatDataset

root = './AllData/competitions/dog-vs-cat-classification'
dataset = DogVsCatDataset(root=root, mode='train')

data, label = dataset[0]
print(data.shape)
print(label)
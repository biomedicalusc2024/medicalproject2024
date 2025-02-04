from dataLoader.segmentation import DataLoader

dataloader = DataLoader(name="CheXmask")
data = dataloader.get_data("df")
print(data)
breakpoint()
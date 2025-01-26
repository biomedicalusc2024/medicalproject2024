from dataLoader.segmentation import DataLoader

dataloader = DataLoader(name="BKAI_IGH")
data = dataloader.get_data("df")
print(data)
breakpoint()
from dataLoader.molecularGeneration import DataLoader

dataloader = DataLoader(name="CrossDocked2020")
data = dataloader.get_data("df")
print(data)
breakpoint()
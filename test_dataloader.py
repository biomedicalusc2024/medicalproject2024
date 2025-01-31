from dataLoader.summerization import DataLoader

dataloader = DataLoader(name="TREC")
data = dataloader.get_data("df")
print(data)
breakpoint()
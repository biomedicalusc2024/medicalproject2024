from dataLoader.questionAnswering import DataLoader

dataloader = DataLoader(name="LLaVA_Med")
data = dataloader.get_data("df")
print(data)
breakpoint()
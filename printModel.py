from model import Generator
from torchsummary import summary

model = Generator()
summary(model, (3, 192, 192))

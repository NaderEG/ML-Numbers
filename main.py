import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import io

import tkinter as tk
from PIL import Image, ImageDraw


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, transform=transform)

loaders = {
    'train': torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),

}

test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(),)

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
    
model = DigitRecognizer().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
total_step = len(loaders['train'])
epochs = 10
for epoch in range(epochs):
    for i, images, labels in enumerate(loaders['train']):
        images, labels = images.to(device), labels.to(device)
        b_x = Variable(images)
        b_y = Variable(labels)
        output = model(b_x)[0]

        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1,total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, in loaders['test']:
        test_output, last_layer = model(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = (pred_y==labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


def recognize_digit():
    with torch.no_grad():
        img = img_canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(img.encode('utf-8')))
        img = img.resize((28, 28)).convert('L')
        img = transform(img)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        result_label.config(text=f"Predicted digit: {predicted.item()}")

root = tk.Tk()
root.title("Handwritten Digit Recognition")

img_canvas = tk.Canvas(root, width=200, height=200, bg="white")
img_canvas.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

img_canvas.bind("<B1-Motion>", lambda event: img_canvas.create_oval(event.x - 10, event.y - 10, event.x + 10, event.y + 10, fill="black"))

recognize_button = tk.Button(root, text="Recognize Digit", command=recognize_digit)
recognize_button.pack()

root.mainloop()
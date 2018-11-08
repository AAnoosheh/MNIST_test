from PIL import Image
import torch
from torchvision import transforms

from options.test_options import TestOptions
from models.networks import CustomNet


def tensor_from_impath(impath, tform):
    try:
        img = Image.open(impath).convert('L')
    except IOError:
        print("Image file not found or format is incorrect")
        return
    return tform(img).unsqueeze(0)


def infer(model, data, device):
    data = data.to(device)
    with torch.no_grad():
        output = model(data).squeeze()
        pred = output.argmax()
        prob = torch.exp(output.max())
    return pred.item(), prob.item()



# Test settings
args = TestOptions().parse()

# Configure GPU
if not args.gpu_id < 0 and torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load pretrained network
model = CustomNet(args).to(device)
model.load_network(args.checkpoints_dir, args.name, args.which_epoch)
model.eval()

# Preprocessing function
tform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])

# Continuous user input
print("""\nWelcome to this MNIST inference server. Please enter valid paths to
         an MNIST image you wish to test, or enter 'Q' to quit\n""")
while True:
    path = input("Please enter a full path to a valid MNIST Image: ")
    if path == 'Q':
        break

    data = tensor_from_impath(path, tform)
    pred, prob = infer(model, data, device)
    print("Predicted digit {} with probability {:.4f}".format(pred, prob))

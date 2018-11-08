import os
import torch
from torchvision import datasets, transforms

from options.train_options import TrainOptions
from models.networks import CustomNet


def load_MNIST_train_data(args, kwargs={}):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))]
    if not args.no_flip:
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list

    mnist_root = os.path.join(args.data_root, 'mnist')
    if not os.path.exists(mnist_root):
        os.makedirs(mnist_root)

    mnist = datasets.MNIST(mnist_root, train=True, download=True,
                transform=transforms.Compose(transform_list))
    return torch.utils.data.DataLoader(mnist,
        batch_size=args.batch_size, shuffle=True, **kwargs)


def train(args, model, dataloader, optimizer, loss_fn, device):
    # Load model if resuming training
    if args.continue_train and args.which_epoch > 0:
        model.load_network(args.checkpoints_dir, args.name, args.which_epoch)

    # Cycle epochs and iters
    for epoch in range(args.which_epoch + 1, args.num_epochs + 1):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        # Log info periodically
        if epoch % args.print_freq == 0:
            print("Train Epoch: {}\tLoss: {:.6f}".format(epoch, loss.item()))

        # Save model checkpoint
        if epoch % args.save_freq == 0:
            model.save_network(args.checkpoints_dir, args.name, epoch)



# Train settings
args = TrainOptions().parse()

# Configure GPU/CPU
if not args.gpu_id < 0 and torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    device = torch.device("cpu")
    kwargs = {'num_workers': args.n_threads}

# Data Loading
train_loader = load_MNIST_train_data(args, kwargs)

# Create network
model = CustomNet(args).to(device)
model.train()

# Optimization modules
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_fn = torch.nn.functional.nll_loss

# Begin training
train(args, model, train_loader, optimizer, loss_fn, device)
print("Training Complete")

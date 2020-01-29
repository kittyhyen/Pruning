import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.nn.init as init
import fc1

import copy

LR = 0.0015
BATCH_SIZE = 60
END_ITER = 5
PRUNE_PERCENT = 50
PRUNE_ITERATIONS = 5
GPU_NUM="1"

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    traindataset = datasets.MNIST('./data', train=True, download=True,transform=transform)
    testdataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=True)

    # model
    model = fc1.fc1().to(device)
    model.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())

    # Making Initial Mask
    mask= make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Print
    for name, param in model.named_parameters():
        print(name, param.size())


    for _ite in range(PRUNE_ITERATIONS):
        best_accuracy = 0
        if not _ite == 0:
            mask= prune_by_percentile(PRUNE_PERCENT, model, mask)

            #model.apply(weight_init)
            #lottery
            original_initialization(mask, initial_state_dict, model)

            ###
            step = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.cpu().numpy()
                                                  * mask[step]).to(weight_dev)
                    step = step + 1
            step = 0
            ###


            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{_ite}/{PRUNE_ITERATIONS}]: ---")
        print_nonzeros(model)

        for iter_ in range(END_ITER):
            # Training
            loss = train(model, train_loader, optimizer, criterion)

            # Testing
            accuracy = test(model, test_loader, criterion)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            print(f'Train Epoch: {iter_}/{END_ITER} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')



   
# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, model, mask):
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            mask[step] = new_mask
            step += 1
    return mask


# Function to make an empty mask of the same size as the model
def make_mask(model):
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    return mask

# Function for Initialization
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero/total)*100,1))


def original_initialization(mask_temp, initial_state_dict, model):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]



if __name__=="__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= GPU_NUM

    main()

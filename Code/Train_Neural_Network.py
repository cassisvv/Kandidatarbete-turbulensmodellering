import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Neural_network import ThePredictionMachine, train_loop, test_loop

#################### Inputs here ##############################
from Process_Hill import X_train, y_train, X_test, y_test, c_DNS_test_dict

DataName = 'Hill'
depth = 2
width = 10
learning_rate = 0.0001
my_batch_size = 10
epochs = 10
batch_normalization = True

#################################################################

def main(depth, width):
    # convert the numpy arrays to PyTorch tensors with float32 data type
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # create PyTorch datasets and dataloaders for the training and validation sets
    # a TensorDataset wraps the feature and target tensors into a single dataset
    # a DataLoader loads the data in batches and shuffles the batches if shuffle=True
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=my_batch_size)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=my_batch_size)

    # --------------------------- Run -----------------------------
    start_time = time.time()

    neural_net = ThePredictionMachine(X_train.shape[1],y_train.shape[1],depth,width,bnorm=batch_normalization)

    loss_fn = nn.MSELoss()

    # Choose loss function, check out https://pytorch.org/docs/stable/optim.html for more info
    # In this case we choose Stocastic Gradient Descent. 
    optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)

    loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, neural_net, loss_fn, optimizer)
        test_loss = test_loop(test_loader, neural_net, loss_fn)
        loss.append(test_loss)
    print("Done!")

    preds = neural_net(X_test_tensor)

    print(f"{'time ML: '}{time.time()-start_time:.2e}")
    import matplotlib.pyplot as plt
    plt.plot([i+1 for i in range(len(loss))],loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    #------------------------results------------------------------------------------------
    c_NN = preds.detach().numpy()
    i = 0
    for c in c_DNS_test_dict:
        print(c+'_error_std', np.std(c_NN[:,i]-c_DNS_test_dict[c])/(np.mean(c_NN[:,i].flatten()**2))**0.5)
        print(c + '_error_RMSE', np.sqrt(sum((c_NN[:, i] - c_DNS_test_dict[c]) ** 2) / len(c_NN[:, i])))
        i += 1

    torch.save(neural_net, '{}depth{}width{}epochs{}batchsize{}rate{}.pt'.format(DataName,depth,width,epochs,my_batch_size,learning_rate))


if __name__ == '__main__':
    main(depth,width)
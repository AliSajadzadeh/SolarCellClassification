import torch
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
    def acc_func(y_predict,real_y):
        _, predicted = t.max(y_predict,1)
        correct_sum = (predicted ==real_y).sum().float()
        accuracy = correct_sum/real_y.shape[0]
        accuracy = torch.round(accuracy*100)
        return accuracy



    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        with t.set_grad_enabled(True):
            outputs = self._model(x)
            loss = self._crit(outputs,y)
            acc = Trainer.acc_func(outputs,y)
            loss.backward()
            self._optim.step()
        return  loss, acc


        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

        self._optim.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = self._model(x)
            loss = self._crit(outputs,y)
            #acc = Trainer.acc_func(outputs,y)
            return loss, outputs
        #TODO
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        running_loss = 0
        running_acc = 0
        self._model.train()
        for i, data in enumerate(self._train_dl,0):
            inputs,labels = data
            batch_loss,batch_acc = Trainer.train_step(inputs,labels)
            running_loss += batch_loss
            running_acc += batch_acc
        return (running_loss/len(self._train_dl)), (running_acc/len(self._train_dl))



        #TODO
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        running_loss_val = 0
        running_acc_val = 0
        self._model.eval()
        for i, data in enumerate(self._val_test_dl, 0):
            inputs, labels = data
            batch_loss, batch_predict = Trainer.val_test_step(inputs, labels)
            running_loss_val += batch_loss
            acc_val = Trainer.acc_func(batch_predict,labels)
            running_acc_val += acc_val
        return (batch_loss / len(self._train_dl)), (acc_val / len(self._train_dl))

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        Epochs = epochs
        train_losses = []
        train_accuracy = []

        val_losses = []
        val_accuracy = []
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(self._optim,mode = 'min',
                                                           factor=0.1,
                                                           patience=10,
                                                           min_lr=1e-5, verbose=False)
        min_val_loss = 10000
        best_epoch = 0
        for epoch in range(Epochs):
            epoch_loss_train, epoch_acc_train = Trainer.train_epoch()
            train_losses.append(epoch_loss_train)
            train_accuracy.append(epoch_acc_train)

            epoch_loss_val, epoch_acc_val = Trainer.val_test()
            val_losses.append(epoch_loss_val)
            val_accuracy.append(epoch_acc_val)

            print("Epoch:{}, training_loss:{:.3f},training_acc:{:.3f},val_loss:{:.3f},val_acc:{:.3f}"
                  .format(epoch+1,epoch_loss_train,epoch_acc_train,epoch_loss_val,epoch_acc_val))

            if epoch_loss_val < min_val_loss:
                min_val_loss = epoch_loss_val
                best_epoch = epoch
                Trainer.save_checkpoint(epoch)
        return train_losses , val_losses






        #TODO
        

      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
        #TODO
                    
        
        
        

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tnrange
from CoinDataset import CoinDataset
import torchvision.transforms as transforms
import pandas as pd 
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class ModelTrainer:
    
    def __init__(self, 
            model,  
            train_path,
            train_embedding_path, 
            val_path,
            val_embedding_path,
            train_augmentations,
            save_path,
            postfix, 
            **kwargs):
        """
        Creates a solver for classification.

        Parameters:
            -train_path (str):
                  The path to the training dataset.
            - train_embedding_path (str):
                  The path to the preprocessed embeddings for the training dataset.
            - val_path (str):
                  The path to the validation dataset.
            - val_embedding_path (str):
                  The path to the preprocessed embeddings for the validation dataset.
            - train_augmentations (torchvision.transforms.transforms.Compose):
                  The data augmentations to apply to the training dataset.
            - save_path : str
                  The path where the trained model will be saved.
            - postfix : str
                  The postfix to append to the saved model's filename.
            - model (nn.Module):
                  Model to be trained.
            - loss (str):
                  Class name of the loss function to be optimized.
                  [Default: 'CrossEntropyLoss']
            - loss_config (dict|None):
                  Dictionary with keyword arguments for calling the loss function.
                  [Default: {}]
            - optimizer (str):
                  Class name of the optimizer to be used.
                  [Default: 'Adam']
            - optimizer_config (dict):
                  Dictionary with keyword arguments for calling for the optimizer.
                  Model parameters don't have to be passed explicitly.
                  [Default: {'lr': 0.0001}]
            - batch_size (int):
                  Number of samples per minibatch.
                  [Default: 16]
            - scheduler (str|None):
                  Class name of the learning rate scheduler to be used.
                  If parameter is not given or `None`, no scheduler is used.
                  [Default: lr_scheduler.StepLR]
            - scheduler_config (dict):
                  Dictionary with keyword arguments to provide for the scheduler.
                  The optimizer is passed in automatically.
                  [Default: {'step_size':4, 'gamma':0.5}]

        """
        # Train on the GPU if possible.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = model.to(self.device)
        # Define default values for parameters.
        defaults = {
            'loss': 'CrossEntropyLoss',
            'loss_config': {},
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.0001},
            'batch_size': 16,
            'scheduler': 'StepLR',
            'scheduler_config': {'step_size':4, 'gamma':0.5}
        }

        # Get given argument or take default value.
        values = defaults | kwargs
        self.train_path = train_path
        self.val_path = val_path
        self.train_embedding_path = train_embedding_path
        self.val_embedding_path = val_embedding_path
        self.train_augmentations = train_augmentations
        
        self.save_path = save_path
        self.postfix = postfix
        self.batch_size = values.pop('batch_size')
        # Store training and validation data.
        self.dataloaders, self.dataset_sizes = self.prepare_dataset()
         

        

        # Create loss function.
        loss = getattr(nn, values.pop('loss'))
        self.loss = loss(**values.pop('loss_config'))

        
        # Create optimizer.
        if values['optimizer'] == 'SophiaG':
            
            values.pop('optimizer')
        else:
            optimizer = getattr(torch.optim, values.pop('optimizer'))
            self.optimizer = optimizer(model.parameters(), **values.pop('optimizer_params'))

        # Scheduler is optional.
        self.scheduler = values.pop('scheduler')

        # Create scheduler if necessary.
        if self.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)
            self.scheduler = scheduler(self.optimizer, **values.pop('scheduler_config'))
        
        
        # Store remaining arguments.
        self.__dict__ |= values
        
        
        
        # Some attributes for bookkeeping.
        self.epoch = 0
        self.num_epochs = 0
        self.val_loss_history = []
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def prepare_dataset(self):
        
        df = pd.read_csv(self.train_path, delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

        unique_classes = df["class"].unique()
        #
        transform = transforms.Compose([
                        transforms.ToTensor(),  # convert images to tensors
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize images
                        transforms.Resize((299, 299))
                    ])
        
        #apply augmentations
        train_dataset = CoinDataset(self.train_path, self.train_embedding_path,
                    transform=self.train_augmentations, num_classes=len(unique_classes))
        val_dataset = CoinDataset(self.val_path, self.val_embedding_path,
                    transform=transform, num_classes=len(unique_classes))
        print(len(train_dataset))
        # Define the dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        

        dataloaders = {"train" : train_dataloader, "val": val_dataloader}
        dataset_sizes = {"train": len(train_dataset), "val" : len(val_dataset)}
        
        for i, batch in enumerate(train_dataloader):
            x, y , z= batch["image"], batch["label"], batch['text_embedding']
            print(x.shape, y.shape, z.shape)
            break
        print("data loaded")
                
        return dataloaders, dataset_sizes

    def save(self, path, safe_model=True):
        """
        Save model and training state to disk. Because we just want to safe the best model we 
        also can save the training stats without the model.

        Parameters:
            - path (str): Path to store checkpoint.

        """
        if safe_model:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'num_epochs': self.num_epochs,
                'val_loss_history': self.val_loss_history,
                'train_loss_history' : self.train_loss_history,
                'train_acc': self.train_acc_history,
                'val_acc': self.val_acc_history
            }
        
            # Save learning rate scheduler state if defined.
            if self.scheduler:
                checkpoint['scheduler'] = self.scheduler.state_dict()
        else:
            checkpoint = {
                'epoch': self.epoch,
                'num_epochs': self.num_epochs,
                'val_loss_history': self.val_loss_history,
                'train_loss_history' : self.train_loss_history,
                'train_acc': self.train_acc_history,
                'val_acc': self.val_acc_history
            }

        # Save checkpoint to disk.
        torch.save(checkpoint, path)


    def load(self, path):
        """
        Load checkpoint from disk.

        Parameters:
            - path (str): Path to checkpoint.

        """
        checkpoint = torch.load(path)

        # Load model and optimizer state.
        self.model.load_state_dict(checkpoint.pop('model'))
        self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

        # Load learning rate scheduler state if defined.
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

        # Load the remaining attributes.
        self.__dict__ |= checkpoint


    def train(self, num_epochs=10, mulitmodal=False):
        torch.backends.cudnn.benchmark = True
        best_acc = 0.0

        for epoch in (pbar := tnrange(num_epochs)):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                all_preds = []
                all_labels = []
                # Iterate over data.
                for idx, batch in enumerate(self.dataloaders[phase]):
                    
                    # Initialize lists to store predictions and labels
                    
                
                    inputs, labels, text_embedding = batch["image"], batch["label"], batch["text_embedding"]
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    if mulitmodal == True:
                        inputs = {'image': inputs, 'text_embedding': text_embedding}

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        #with torch.autocast(device_type='cuda', dtype=torch.float16):

                            # Forward pass
                        outputs = self.model(inputs)
                            #assert outputs.dtype is torch.float16
                        loss = self.loss(outputs, labels.float())
                            #assert loss.dtype is torch.float32

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            #scaler.step(optimizer)
                            #scaler.update()

                        _, preds = torch.max(outputs, 1)
                        _,labels = torch.max(labels, 1)
                    # statistics
                    if mulitmodal == True:
                        running_loss += loss.item() * inputs['image'].size(0)
                    else:
                        running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                    
                     # Store predictions and labels
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                if phase == 'val':
                    self.val_acc_history.append(epoch_acc)
                    self.val_loss_history.append(epoch_loss)
                else:
                    self.train_acc_history.append(epoch_acc)
                    self.train_loss_history.append(epoch_loss)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                # Calculate and print confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                #print(f'{phase} Confusion Matrix: \n{cm}')
                
                # Visualization of the Confusion Matrix
                
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    
                    self.save(self.save_path+"\\model"+self.
                                      postfix+".tar")
            
            
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        plt.title(f'{phase} Confusion Matrix')
        plt.show()
        print()
        self.save(self.save_path+"\\stats"+self.
                                      postfix+".tar", safe_model=False)
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        
        return self.model
    
    

    def evaluate(self, multimodal=False, show=True):
        def accuracy(output, target, topk=(5,)):
            """Computes the precision@k for the specified values of k"""
            maxk = max(topk)

            _, pred = output.topk(maxk, 1, True, True)
            
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0))
                
            return res
        
        
        
        running_corrects = 0
        top5_corrects = 0

        self.model.eval()
        # Iterate over validation data.
        for idx, batch in enumerate(self.dataloaders['val']):
            inputs, labels, text_embedding = batch["image"], batch["label"], batch["text_embedding"]
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if multimodal == True:
                inputs = {'image': inputs, 'text_embedding': text_embedding}

            # forward
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                _,labels = torch.max(labels, 1)

            # statistics
            running_corrects += torch.sum(preds == labels)
            
            # Compute top-5 accuracy
            top5_corrects += accuracy(outputs, labels)[0].item()
            
        acc = running_corrects.double() / self.dataset_sizes['val']
        top5_acc = top5_corrects / self.dataset_sizes['val']
        if show == True:
            print(f'Validation Top-1 Accuracy: {acc:.4f}')
            print(f'Validation Top-5 Accuracy: {top5_acc:.4f}')
        return acc, top5_acc
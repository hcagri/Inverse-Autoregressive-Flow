import torch 
import copy 
from tqdm.auto import tqdm
from .models import VAE 
from .loss import vae_loss 
from .data import PointDataset
from torch.utils.data import DataLoader


def train(model_params, hparams, device='cpu'):
    
    model = VAE(model_params)
    optim = torch.optim.Adam(model.parameters(), lr = hparams['lr'])
    loss_fn = vae_loss

    train_loader = DataLoader(PointDataset(hparams['train_path']), batch_size=hparams['batch_size'], shuffle=True)
    test_loader = DataLoader(PointDataset(hparams['test_path']), batch_size=hparams['batch_size'], shuffle=True)

    step = 0
    min_error = torch.inf

    avg_loss_train_lst = []
    avg_loss_val_lst = []
    iterator = tqdm(range(1,hparams['epochs']+1), leave=True)
    for epoch in iterator:
        
        iterator.set_description_str(f"Epoch: {epoch}")

        avg_loss = 0
        num_batch = 0

        for point_batch in train_loader:    
            model.train()

            optim.zero_grad()

            point_batch = point_batch.to(device)

            x_recons, log_p_zt, l = model(point_batch)

            loss = loss_fn(point_batch, x_recons, log_p_zt, l)

            loss.backward()
            optim.step()
            step += 1

            avg_loss += loss/point_batch.shape[0]
            num_batch+=1

        avg_loss_train = avg_loss/num_batch

        with torch.no_grad():

            model.eval()
            avg_loss_val = 0
            num_batch_val = 0
            for point_batch in test_loader:
                
                x_recons, log_p_zt, l = model(point_batch)
                loss = loss_fn(point_batch, x_recons, log_p_zt, l)

                avg_loss_val += loss/point_batch.shape[0]
                num_batch_val+=1

            avg_loss_val = avg_loss_val/num_batch_val

            if avg_loss_val < min_error:
                min_error = avg_loss_val
                best_model = copy.deepcopy(model)
        
        iterator.set_postfix_str(f"Average train loss: {avg_loss_train.item():.4f}  Average val loss: {avg_loss_val.item():.4f}")
        avg_loss_train_lst.append(avg_loss_train.item())
        avg_loss_val_lst.append(avg_loss_val.item())
    
    return avg_loss_train_lst, avg_loss_val_lst, best_model
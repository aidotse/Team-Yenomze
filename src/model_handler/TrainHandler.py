import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch import autograd

from torchio.transforms import CropOrPad
from monai.data import ArrayDataset, DataLoader, PILReader
from monai.transforms import Compose, LoadImage, RandFlip, RandRotate, RandRotate90, ToTensor

from src.dataloader.TrainDataset import OurDataset
from src.dataloader.ValidationDataset import OurGridyDataset
from src.loss.VGGLoss import VGGLoss
from src.model.Discriminator import Discriminator
from src.model.Generator import GeneratorUnet
from src.util.DataUtils import (
    split_train_val,
    MozartTheComposer
)

def start_training(batch_size=16, 
                   num_epoch=500,
                   num_epoch_pretrain_G=None,
                   lr=1e-5,
                   unet_split_mode=True,
                   data_dir = "/data/*",
                   load_weight_dir=None,
                   save_weight_dir="./checkpoints/",
                   log_dir="./logs/",
                   loss_dir="./lossinfo/",
                   augmentation_prob=50,
                   adversarial_weight=5e-2,
                   mse_loss_weight=50,
                   c01_weight=0.3,
                   c02_weight=0.3,
                   c03_weight=0.4,
                   is_val_split=False,
                   ):
    if not os.path.exists(save_weight_dir):
        os.makedirs(save_weight_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(f'using GPU? : {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}', )

    writer = SummaryWriter(log_dir=log_dir)  # tensorboard

    # collect instances
    inputs = [
        sorted(glob.glob(os.path.join(data_dir, f'*A04Z0{i}*.tif'), recursive=True))
        for i in range(1,8)
    ]
    targets = [
        sorted(glob.glob(os.path.join(data_dir, f'*C0{i}.tif'), recursive=True))
        for i in range(1,4)
    ]
    # merge inputs and targets
    all_data = inputs + targets
    # match the slices and match all of the data for one input instance
    data_all_ch = list(zip(*all_data))
    
    train_split, val_split = split_train_val(data_all_ch, 
                                             N_valid_per_magn=4,
                                             is_val_split = is_val_split)

    # data preprocessing/augmentation
    trans_train = MozartTheComposer(
            [
                RandRotate(range_x=15, prob=augmentation_prob, keep_size=True, padding_mode="reflection"),
                RandRotate90(prob=augmentation_prob, spatial_axes=(1, 2)),
                RandFlip(spatial_axis=(1, 2), prob=augmentation_prob),
                ToTensor()
            ]
        )

    trans_val = MozartTheComposer(
        [
            ToTensor()
        ]
    )

    # create dataset class
    train_dataset = OurDataset(
        data=train_split,
        data_reader=PILReader(),
        transform=trans_train,
        roi_size=256, samples_per_image=8
    )

    val_dataset = OurGridyDataset(
        data=val_split,
        data_reader=PILReader(),
        patch_size=256
    )

    # now create data loader ( MONAI DataLoader)
    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8 #multiprocessing.cpu_count(),
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4 #multiprocessing.cpu_count(),
    )

    # load model / criterion / optimizer
    netG = GeneratorUnet(split=unet_split_mode).to(device)
    # print(netG)
    netD = Discriminator().to(device)

    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()
    vggloss = VGGLoss(device=device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    optimizerD = optim.Adam(netD.parameters(), lr=lr)

    # load weight if load_weight_dir is defined
    if load_weight_dir is not None:
        print(f'Loading checkpoint: {load_weight_dir}')
        checkpoint = torch.load(load_weight_dir)
        netG.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        init_epoch = 0

    if num_epoch_pretrain_G is not None:
        print(f'pre-training Generator for {num_epoch_pretrain_G} epochs')
        save_gene = pd.DataFrame(columns=['TotalLoss', 'lossC01', 'lossC02', 'lossC03'])

        for epoch in range(1, num_epoch_pretrain_G + 1):
            pretrainedG_losses = []
            lossC01s = []
            lossC02s = []
            lossC03s = []

            netG.train()
            
            # torch.Size([8, 1, 10, 256, 256])
            for batch_index, batch in enumerate(tqdm(training_loader, file=sys.stdout)):
                inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07 = \
                    batch[:,:,0,:,:].to(device), \
                    batch[:,:,1,:,:].to(device), \
                    batch[:,:,2,:,:].to(device), \
                    batch[:,:,3,:,:].to(device), \
                    batch[:,:,4,:,:].to(device), \
                    batch[:,:,5,:,:].to(device), \
                    batch[:,:,6,:,:].to(device)
                targetC01, targetC02, targetC03 = batch[:,:,7,:,:].to(device), \
                    batch[:,:,8,:,:].to(device), \
                    batch[:,:,9,:,:].to(device)

                netG.zero_grad()

                outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07)

                # with autograd.detect_anomaly():
                loss01 = mseloss(outputC01, targetC01)
                loss02 = mseloss(outputC02, targetC02)
                loss03 = mseloss(outputC03, targetC03)

                loss_gene = c01_weight*loss01 + c02_weight*loss02 + c03_weight*loss03
                loss_gene.backward()
                optimizerG.step()

                pretrainedG_losses.append(loss_gene.detach().item())
                lossC01s.append(loss01.detach().item())
                lossC02s.append(loss02.detach().item())
                lossC03s.append(loss03.detach().item())
                
                
            epoch_loss = np.array(pretrainedG_losses).mean()
            epoch_lossC01 = np.array(lossC01s).mean()
            epoch_lossC02 = np.array(lossC02s).mean()
            epoch_lossC03 = np.array(lossC03s).mean()

            print(f'Pre-training Generator - Epoch {epoch}/{num_epoch_pretrain_G} loss: {epoch_loss}, loss01: {epoch_lossC01}, loss02: {epoch_lossC02}, loss03: {epoch_lossC03}')
            save_gene = save_gene.append({'TotalLoss': epoch_loss, 'lossC01': epoch_lossC01, 'lossC02': epoch_lossC02, 'lossC03': epoch_lossC03}, ignore_index=True)
            save_gene.to_csv(os.path.join(loss_dir, 'generator_loss_info.csv'))
            
            # save images
            if epoch % 1 == 0:
                writer.add_images('Groundtruth_G_only/C01', targetC01, epoch)
                writer.add_images('Groundtruth_G_only/C02', targetC02, epoch)
                writer.add_images('Groundtruth_G_only/C03', targetC03, epoch)
                writer.add_images('train_G_only/C01', outputC01, epoch)
                writer.add_images('train_G_only/C02', outputC02, epoch)
                writer.add_images('train_G_only/C03', outputC03, epoch) 

            # save model parameters
            weight = f'pretrained_G_epoch_{epoch}.pth'
            torch.save(netG.state_dict(), os.path.join(save_weight_dir, weight))
            torch.save({'epoch': epoch,
                        'model_state_dict': netG.state_dict(),
                        'optimizer_state_dict': optimizerG.state_dict(),
                        'loss': epoch_loss}, os.path.join(save_weight_dir, weight))
                  
    if num_epoch_pretrain_G is not None:
        print(f'Loading weights from pretrained generator')
        finalweight = f'pretrained_G_epoch_{num_epoch_pretrain_G}.pth'
        checkpoint = torch.load(os.path.join(save_weight_dir, finalweight))
        netG.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        pass


    if num_epoch_pretrain_G is not None:
        init_epoch = num_epoch_pretrain_G + 1
        num_epoch = num_epoch + num_epoch_pretrain_G
    else:
        num_epoch = num_epoch

        if load_weight_dir is None:
            init_epoch = 1
        else:
            init_epoch = init_epoch
            num_epoch = num_epoch + init_epoch

    save_gan_train = pd.DataFrame(columns=['TotalLoss', 'v_lossC01', 'v_lossC02', 'v_lossC03','m_lossC01', 'm_lossC02', 'm_lossC03', 'adv' ,'valTotalLoss',
                                           'val_v_lossC01', 'val_v_lossC02', 'val_v_lossC03','val_m_lossC01', 'val_m_lossC02', 'val_m_lossC03', 'val_adv'])

    for epoch in range(init_epoch, num_epoch + 1):
        print(f'Epoch : [{epoch} / {num_epoch}]')
        netG.train()
        G_losses = []
        D_losses = []

        v_lossC01s = []
        v_lossC02s = []
        v_lossC03s = []
        m_lossC01s = []
        m_lossC02s = []
        m_lossC03s = []

        for batch_index, batch in enumerate(tqdm(training_loader, file=sys.stdout)):
            inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07 = \
                    batch[:,:,0,:,:].to(device), \
                    batch[:,:,1,:,:].to(device), \
                    batch[:,:,2,:,:].to(device), \
                    batch[:,:,3,:,:].to(device), \
                    batch[:,:,4,:,:].to(device), \
                    batch[:,:,5,:,:].to(device), \
                    batch[:,:,6,:,:].to(device)
            targetC01, targetC02, targetC03 = batch[:,:,7,:,:].to(device), \
                    batch[:,:,8,:,:].to(device), \
                    batch[:,:,9,:,:].to(device)

            shape = inputZ01.size()
            real_label = torch.ones((shape[0])).to(device)
            fake_label = torch.zeros((shape[0])).to(device)

            ###############################################################
            # First train discriminator network : maximize D(x)-1-D(G(z)) #
            ###############################################################

            netD.zero_grad()

            outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07)

            targetCs = torch.cat((targetC01, targetC02, targetC03), dim=1)
            realCs_prob = netD(targetCs)

            outputCs = torch.cat((outputC01, outputC02, outputC03), dim=1)
            fakeCs_prob = netD(outputCs)

            d_loss_real = bceloss(realCs_prob, real_label)
            d_loss_fake = bceloss(fakeCs_prob, fake_label)

            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            optimizerD.step()

            D_losses.append(d_loss.item())

            #################################################################
            # Now train Generator network                                   #
            # option 1 (mode='MSE') : minimize mseloss + 10^-3 * -logD(G(z))#
            # option 2 (mode='VGG') : minimize vggloss + 10^-3 * -logD(G(z))#
            #################################################################
            netG.zero_grad()

            outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07)

            
            # calculating MSE loss
            m_lossC01 = mseloss(outputC01, targetC01)
            m_lossC02 = mseloss(outputC02, targetC02)
            m_lossC03 = mseloss(outputC03, targetC03)
            m_content_loss = c01_weight*m_lossC01 + c02_weight*m_lossC02 + c03_weight*m_lossC03

            # calculating VGG loss
            targetC01_vgg = torch.cat((targetC01, targetC01, targetC01), dim=1)  # concat
            targetC02_vgg = torch.cat((targetC02, targetC02, targetC02), dim=1)
            targetC03_vgg = torch.cat((targetC03, targetC03, targetC03), dim=1)
            outputC01_vgg = torch.cat((outputC01,outputC01,outputC01), dim=1)  
            outputC02_vgg = torch.cat((outputC02,outputC02,outputC02), dim=1)
            outputC03_vgg = torch.cat((outputC03,outputC03,outputC03), dim=1)
            
            v_lossC01 = vggloss(outputC01_vgg, targetC01_vgg)
            v_lossC02 = vggloss(outputC02_vgg, targetC02_vgg)
            v_lossC03 = vggloss(outputC03_vgg, targetC03_vgg)
            v_content_loss = c01_weight*v_lossC01 + c02_weight*v_lossC02 + c03_weight*v_lossC03
            
            # calculating weighted content loss
            content_loss = mse_loss_weight * m_content_loss + v_content_loss
        
            outputCs = torch.cat((outputC01, outputC02, outputC03), dim=1)
            fakeCs_prob = netD(outputCs)

            adversarial_loss = bceloss(fakeCs_prob, real_label)

            g_loss = content_loss + adversarial_weight * adversarial_loss
            

            g_loss.backward()
            optimizerG.step()

            G_losses.append(g_loss.detach().item())
            v_lossC01s.append(v_lossC01.detach().item())
            v_lossC02s.append(v_lossC02.detach().item())
            v_lossC03s.append(v_lossC03.detach().item())
            m_lossC01s.append(m_lossC01.detach().item())
            m_lossC02s.append(m_lossC02.detach().item())
            m_lossC03s.append(m_lossC03.detach().item())


            # log to tensorboard every 10 steps
            if batch_index % 10 == 0:
                writer.add_scalar('Train/G_loss', g_loss.item(), epoch)
                writer.add_scalar('Train/D_loss', d_loss.item(), epoch)

        
        if epoch % 1 == 0:
            writer.add_images('Groundtruth_train/C01', targetC01, epoch)
            writer.add_images('Groundtruth_train/C02', targetC02, epoch)
            writer.add_images('Groundtruth_train/C03', targetC03, epoch)
            writer.add_images('train/C01', outputC01, epoch)
            writer.add_images('train/C02', outputC02, epoch)
            writer.add_images('train/C03', outputC03, epoch)
            
        G_loss = np.array(G_losses).mean()
        D_loss = np.array(D_losses).mean()
        epoch_v_lossC01 = np.array(v_lossC01s).mean()
        epoch_v_lossC02 = np.array(v_lossC02s).mean()
        epoch_v_lossC03 = np.array(v_lossC03s).mean()
        epoch_m_lossC01 = np.array(m_lossC01s).mean()
        epoch_m_lossC02 = np.array(m_lossC02s).mean()
        epoch_m_lossC03 = np.array(m_lossC03s).mean()

        print(f'Epoch {epoch}/{num_epoch} Training g_loss : {G_loss}, d_loss : {D_loss}')

        ################################################################################################################
        # Validation
        netG.eval()

        val_G_losses = []
        val_v_lossC01s = []
        val_v_lossC02s = []
        val_v_lossC03s = []
        val_m_lossC01s = []
        val_m_lossC02s = []
        val_m_lossC03s = []

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(validation_loader, file=sys.stdout)):
                inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07 = \
                    batch[0][:,:,0,:,:].to(device), \
                    batch[0][:,:,1,:,:].to(device), \
                    batch[0][:,:,2,:,:].to(device), \
                    batch[0][:,:,3,:,:].to(device), \
                    batch[0][:,:,4,:,:].to(device), \
                    batch[0][:,:,5,:,:].to(device), \
                    batch[0][:,:,6,:,:].to(device)
                targetC01, targetC02, targetC03 = batch[0][:,:,7,:,:].to(device), \
                    batch[0][:,:,8,:,:].to(device), \
                    batch[0][:,:,9,:,:].to(device)

                shape = inputZ01.size()
                real_label = torch.ones((shape[0])).to(device)
                fake_label = torch.zeros((shape[0])).to(device)

                outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07)

                # calculating MSE loss
                val_m_lossC01 = mseloss(outputC01, targetC01)
                val_m_lossC02 = mseloss(outputC02, targetC02)
                val_m_lossC03 = mseloss(outputC03, targetC03)
                val_m_content_loss = c01_weight*val_m_lossC01 + c02_weight*val_m_lossC02 + c03_weight*val_m_lossC03

                # calculating VGG loss
                targetC01_vgg = torch.cat((targetC01, targetC01, targetC01), dim=1)
                targetC02_vgg = torch.cat((targetC02, targetC02, targetC02), dim=1)
                targetC03_vgg = torch.cat((targetC03, targetC03, targetC03), dim=1)
                outputC01_vgg = torch.cat((outputC01,outputC01,outputC01), dim=1)
                outputC02_vgg = torch.cat((outputC02,outputC02,outputC02), dim=1)
                outputC03_vgg = torch.cat((outputC03,outputC03,outputC03), dim=1)
                
                val_v_lossC01 = vggloss(outputC01_vgg, targetC01_vgg)
                val_v_lossC02 = vggloss(outputC02_vgg, targetC02_vgg)
                val_v_lossC03 = vggloss(outputC03_vgg, targetC03_vgg)
                val_v_content_loss = c01_weight*val_v_lossC01 + c02_weight*val_v_lossC02 + c03_weight*val_v_lossC03
                
                # calculating weighted content loss
                val_content_loss = mse_loss_weight * val_m_content_loss + val_v_content_loss

                outputCs = torch.cat((outputC01, outputC02, outputC03), dim=1)
                fakeCs_prob = netD(outputCs)

                val_adversarial_loss = bceloss(fakeCs_prob, real_label)

                val_g_loss = val_content_loss + adversarial_weight * val_adversarial_loss

                val_G_losses.append(val_g_loss.detach().item())
                val_v_lossC01s.append(val_v_lossC01.detach().item())
                val_v_lossC02s.append(val_v_lossC02.detach().item())
                val_v_lossC03s.append(val_v_lossC03.detach().item())
                val_m_lossC01s.append(val_m_lossC01.detach().item())
                val_m_lossC02s.append(val_m_lossC02.detach().item())
                val_m_lossC03s.append(val_m_lossC03.detach().item())


                # log to tensorboard every 10 steps
                if batch_index % 10 == 0:
                    writer.add_scalar('Val/G_loss', val_g_loss.item(), epoch)
        
        if epoch % 1 == 0:
            writer.add_images('Groundtruth_val/C01', targetC01, epoch)
            writer.add_images('Groundtruth_val/C02', targetC02, epoch)
            writer.add_images('Groundtruth_val/C03', targetC03, epoch)
            writer.add_images('val/C01', outputC01, epoch)
            writer.add_images('val/C02', outputC02, epoch)
            writer.add_images('val/C03', outputC03, epoch)          
                  
        val_G_loss = np.array(val_G_losses).mean()
        val_epoch_v_lossC01 = np.array(val_v_lossC01s).mean()
        val_epoch_v_lossC02 = np.array(val_v_lossC02s).mean()
        val_epoch_v_lossC03 = np.array(val_v_lossC03s).mean()
        val_epoch_m_lossC01 = np.array(val_m_lossC01s).mean()
        val_epoch_m_lossC02 = np.array(val_m_lossC02s).mean()
        val_epoch_m_lossC03 = np.array(val_m_lossC03s).mean()

        save_gan_train = save_gan_train.append(
            {'TotalLoss': G_loss, 'v_lossC01': epoch_v_lossC01, 'v_lossC02': epoch_v_lossC02, 'v_lossC03': epoch_v_lossC03, 
             'm_lossC01': epoch_m_lossC01, 'm_lossC02': epoch_m_lossC02, 'm_lossC03': epoch_m_lossC03, 'adv': adversarial_loss, 
             'valTotalLoss': val_G_loss, 'val_v_lossC01': val_epoch_v_lossC01, 'val_v_lossC02': val_epoch_v_lossC02, 'val_v_lossC03': val_epoch_v_lossC03,
             'val_m_lossC01': val_epoch_m_lossC01, 'val_m_lossC02': val_epoch_m_lossC02, 'val_m_lossC03': val_epoch_m_lossC03, 'val_adv': val_adversarial_loss},
            ignore_index=True)
        save_gan_train.to_csv(os.path.join(loss_dir, 'gan_train_loss_info.csv')) 

        print(f'Epoch {epoch}/{num_epoch} Validation g_loss : {val_G_loss}')

        # now save model parameter (from training)
        weight_g = f'G_epoch_{epoch}.pth'
        torch.save({'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    'loss': G_loss}, os.path.join(save_weight_dir, weight_g))

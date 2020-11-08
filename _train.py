import argparse
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
import multiprocessing

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch import autograd

from torchio.transforms import CropOrPad
from monai.data import ArrayDataset, DataLoader, PILReader
from monai.transforms import Compose, LoadImage, AddChannel, RandFlip, RandRotate, RandRotate90, RandScaleIntensity, CenterSpatialCrop, ToTensor, ScaleIntensity, LoadPNG
from monai.visualize import plot_2d_or_3d_image

import FlowArrayDataset

from utils import *
from VGGLoss import *
from Generator import *
from Discriminator import *

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--num_epochs_G', default=10, type=int,
                    help='number of epochs to pre-train generator, otherwise set to None')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--mode', default='VGG', type=str,
                    help='MSE for MSE as a content loss or VGG for pretrained vgg19 as a content loss')
parser.add_argument('--aug_prob', default=15, type=int, help='augmentation probability')
parser.add_argument('--data_dir', default='./images/20x_images/*', type=str) #'./images/*/*'
parser.add_argument('--load_weight_dir', default=None, type=str,
                    help='if you want to continue training load the checkpoint, otherwise set to None')
parser.add_argument('--save_weight_dir', default='./checkpoints/tempcheckpoint',
                    type=str, help='directory where training weightes are saved')
parser.add_argument('--log_dir', default='./logs/templog',
                    type=str, help='directory where tensorboard logs are saved')
parser.add_argument('--save_loss_dir', default='./lossinfo/templossinfo',
                    type=str, help='directory for loss information')

if __name__ == '__main__':
    arg = parser.parse_args()

    num_epochs_G = arg.num_epochs_G
    num_epochs = arg.num_epochs
    batch_size = arg.batch_size
    mode = arg.mode
    aug_prob = arg.aug_prob
    data_dir = arg.data_dir
    load_weight_dir = arg.load_weight_dir
    save_weight_dir = arg.save_weight_dir
    log_dir = arg.log_dir
    loss_dir = arg.save_loss_dir

    if not os.path.exists(save_weight_dir):
        os.makedirs(save_weight_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('using GPU? : ', torch.cuda.is_available())

    writer = SummaryWriter(log_dir=log_dir)  # tensorboard

    # 6 input slices
    inputZ01_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z01*.tif'), recursive=True))
    inputZ02_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z02*.tif'), recursive=True))
    inputZ03_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z03*.tif'), recursive=True))
    inputZ04_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z04*.tif'), recursive=True))
    inputZ05_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z05*.tif'), recursive=True))
    inputZ06_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z06*.tif'), recursive=True))
    inputZ07_path = sorted(glob.glob(os.path.join(data_dir, '*A04Z07*.tif'), recursive=True))

    # 3 output channels
    targetC01_path = sorted(glob.glob(os.path.join(data_dir, '*C01.tif'), recursive=True))
    targetC02_path = sorted(glob.glob(os.path.join(data_dir, '*C02.tif'), recursive=True))
    targetC03_path = sorted(glob.glob(os.path.join(data_dir, '*C03.tif'), recursive=True))

    # split training/validation
    inputZ01, inputZ01_val = split_train_val(inputZ01_path)
    inputZ02, inputZ02_val = split_train_val(inputZ02_path)
    inputZ03, inputZ03_val = split_train_val(inputZ03_path)
    inputZ04, inputZ04_val = split_train_val(inputZ04_path)
    inputZ05, inputZ05_val = split_train_val(inputZ05_path)
    inputZ06, inputZ06_val = split_train_val(inputZ06_path)
    inputZ07, inputZ07_val = split_train_val(inputZ07_path)

    targetC01, targetC01_val = split_train_val(targetC01_path)
    targetC02, targetC02_val = split_train_val(targetC02_path)
    targetC03, targetC03_val = split_train_val(targetC03_path)

    # data preprocessing/augmentation
    trans_train = Compose(
        [
            #LoadPNG(image_only=True),
            LoadImage(PILReader(), image_only=True),
            AddChannel(),
            CenterSpatialCrop(roi_size=2154),  # 2154
            #ScaleIntensity(),
            #RandRotate(range_x=15, prob=aug_prob, keep_size=True),
            #RandRotate90(prob=aug_prob, spatial_axes=(0, 1)),
            #RandFlip(spatial_axis=0, prob=aug_prob),
            #RandScaleIntensity(factors=0.5, prob=aug_prob)
            ToTensor()
        ]
    )

    trans_val = Compose(
        [
            #LoadPNG(image_only=True),
            LoadImage(PILReader(), image_only=True),
            AddChannel(),
            #CenterSpatialCrop(roi_size=2154),
            #ScaleIntensity(),
            ToTensor()
        ]
    )

    # create dataset class
    train_dataset = FlowArrayDataset.FlowArrayDataset(
        inputZ01=inputZ01, inputZ01_transform=trans_train,
        inputZ02=inputZ02, inputZ02_transform=trans_train,
        inputZ03=inputZ03, inputZ03_transform=trans_train,
        inputZ04=inputZ04, inputZ04_transform=trans_train,
        inputZ05=inputZ05, inputZ05_transform=trans_train,
        inputZ06=inputZ06, inputZ06_transform=trans_train,
        inputZ07=inputZ07, inputZ07_transform=trans_train,
        targetC01=targetC01, targetC01_transform=trans_train,
        targetC02=targetC02, targetC02_transform=trans_train,
        targetC03=targetC03, targetC03_transform=trans_train
    )

    val_dataset = FlowArrayDataset.FlowArrayDataset(
        inputZ01=inputZ01_val, inputZ01_transform=trans_val,
        inputZ02=inputZ02_val, inputZ02_transform=trans_val,
        inputZ03=inputZ03_val, inputZ03_transform=trans_val,
        inputZ04=inputZ04_val, inputZ04_transform=trans_val,
        inputZ05=inputZ05_val, inputZ05_transform=trans_val,
        inputZ06=inputZ06_val, inputZ06_transform=trans_val,
        inputZ07=inputZ07_val, inputZ07_transform=trans_val,
        targetC01=targetC01_val, targetC01_transform=trans_val,
        targetC02=targetC02_val, targetC02_transform=trans_val,
        targetC03=targetC03_val, targetC03_transform=trans_val
    )

    # now create data loader ( MONAI DataLoader)

    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size
        #shuffle=True
        #num_workers=multiprocessing.cpu_count(),
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
        #num_workers=multiprocessing.cpu_count(),
    )

    # load model / criterion / optimizer
    netG = GeneratorUnet().to(device)
    print(netG)
    netD = Discriminator().to(device)

    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()
    vggloss = VGGLoss()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

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


    ############################
    # pre-train generator only #
    ############################

    if num_epochs_G is not None:
        print(f'pre-training Generator for {num_epochs_G} epochs')
        save_gene = pd.DataFrame(columns=['TotalLoss', 'lossC01', 'lossC02', 'lossC03'])

        for epoch in range(1, num_epochs_G + 1):
            pretrainedG_losses = []
            lossC01s = []
            lossC02s = []
            lossC03s = []

            netG.train()

            for batch_index, batch in enumerate(tqdm(training_loader)):
                inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07 = \
                    batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                    batch[3].to(device), batch[4].to(device), batch[5].to(device), batch[6].to(device)
                targetC01, targetC02, targetC03 = batch[7].to(device), batch[8].to(device), batch[9].to(device)

                print('input size :' + str(inputZ01.size()))
                print('target size :' + str(targetC01.size()))

                netG.zero_grad()

                outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07)

                # with autograd.detect_anomaly():
                loss01 = mseloss(outputC01, targetC01)
                loss02 = mseloss(outputC01, targetC01)
                loss03 = mseloss(outputC01, targetC01)

                loss_gene = loss01 + loss02 + loss03
                loss_gene.backward()
                optimizerG.step()

                pretrainedG_losses.append(loss_gene.item())
                lossC01s.append(loss01.item())
                lossC02s.append(loss02.item())
                lossC03s.append(loss03.item())

            epoch_loss = np.array(pretrainedG_losses).mean()
            epoch_lossC01 = np.array(lossC01s).mean()
            epoch_lossC02 = np.array(lossC02s).mean()
            epoch_lossC03 = np.array(lossC03s).mean()

            print(f'Pre-training Generator - Epoch {epoch}/{num_epochs} loss: {epoch_loss}, loss01: {epoch_lossC01}, loss02: {epoch_lossC02}, loss03: {epoch_lossC03}')
            save_gene = save_gene.append({'TotalLoss': epoch_loss, 'lossC01': epoch_lossC01, 'lossC02': epoch_lossC02, 'lossC03': epoch_lossC03}, ignore_index=True)
            save_gene.to_csv(os.path.join(loss_dir, 'generator_loss_info.csv'))

            # save model parameters
            weight = f'pretrained_G_epoch_{epoch}.pth'
            torch.save(netG.state_dict(), os.path.join(save_weight_dir, weight))

    if num_epochs_G is not None:
        init_epoch = num_epochs_G + 1
        num_epochs = num_epochs + num_epochs_G
    else:
        init_epoch = 1
        num_epochs = num_epochs

    # train GAN

    real_label = torch.ones((batch_size, 1)).to(device)
    fake_label = torch.zeros((batch_size, 1)).to(device)
    # OR....use Label smoothing
    # real_label = torch.tensor(torch.rand(real_prob.size()) * 0.25 + 0.85)
    # fake_label = torch.tensor(torch.rand(fake_prob.size()) * 0.15)

    print(f'\nnow training GAN')
    save_gan_train = pd.DataFrame(columns=['TotalLoss', 'lossC01', 'lossC02', 'lossC03'])
    save_gan_val = pd.DataFrame(columns=['TotalLoss', 'lossC01', 'lossC02', 'lossC03'])

    for epoch in range(init_epoch, num_epochs + 1):
        print(f'\nEpoch : [{epoch} / {num_epochs}]')
        netG.train()
        G_losses = []
        D_losses = []

        lossC01s = []
        lossC02s = []
        lossC03s = []

        for batch_index, batch in enumerate(tqdm(training_loader)):
            inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06 = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device),\
                batch[3].to(device), batch[4].to(device), batch[5].to(device)
            targetC01, targetC02, targetC03 = batch[6].to(device), batch[7].to(device), batch[8].to(device)

            ###############################################################
            # First train discriminator network : maximize D(x)-1-D(G(z)) #
            ###############################################################

            netD.zero_grad()

            realC01_prob = netD(targetC01)
            realC02_prob = netD(targetC02)
            realC03_prob = netD(targetC03)

            # torch.cat((targetC01, targetC02, targetC03), dim=1) and pass it to disc

            outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06, inputZ07)
            fakeC01_prob = netD(outputC01)
            fakeC02_prob = netD(outputC02)
            fakeC03_prob = netD(outputC03)

            d_loss_real = bceloss(realC01_prob, real_label) + bceloss(realC02_prob, real_label) + bceloss(realC03_prob, real_label)
            d_loss_fake = bceloss(fakeC01_prob, fake_label) + bceloss(fakeC02_prob, fake_label) + bceloss(fakeC02_prob, fake_label)

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

            if mode == 'MSE':
                lossC01 = mseloss(outputC01, targetC01)
                lossC02 = mseloss(outputC02, targetC02)
                lossC03 = mseloss(outputC03, targetC03)
                content_loss = lossC01 + lossC02 + lossC03

            if mode == 'VGG':
                lossC01 = vggloss(outputC01, targetC01)
                lossC02 = vggloss(outputC02, targetC02)
                lossC03 = vggloss(outputC03, targetC03)
                content_loss = lossC01 + lossC02 + lossC03

            adversarial_loss = bceloss(netD(outputC01), real_label) + bceloss(netD(outputC02), real_label) + bceloss(netD(outputC03), real_label)
            # can try bce loss on bceloss(outputC1-3, targetC1-3)

            g_loss = content_loss + 1e-3 * adversarial_loss

            g_loss.backward()
            optimizerG.step()

            G_losses.append(g_loss.item())
            lossC01s.append(lossC01.item())
            lossC02s.append(lossC02.item())
            lossC03s.append(lossC03.item())


            # log to tensorboard every 10 steps
            if batch_index % 1 == 0:
                writer.add_scalar('Train/G_loss', g_loss.item(), epoch)
                writer.add_scalar('Train/D_loss', d_loss.item(), epoch)
                plot_2d_or_3d_image(targetC01, epoch * len(training_loader) + batch_index, writer, index=0, tag="Groundtruth_train/C01")
                plot_2d_or_3d_image(targetC02, epoch * len(training_loader) + batch_index, writer, index=0, tag="Groundtruth_train/C01")
                plot_2d_or_3d_image(targetC03, epoch * len(training_loader) + batch_index, writer, index=0, tag="Groundtruth_train/C01")
                plot_2d_or_3d_image(outputC01, epoch * len(training_loader) + batch_index, writer, index=0, tag="Train/C01")
                plot_2d_or_3d_image(outputC02, epoch * len(training_loader) + batch_index, writer, index=0, tag="Train/C01")
                plot_2d_or_3d_image(outputC03, epoch * len(training_loader) + batch_index, writer, index=0, tag="Train/C01")

        G_loss = np.array(G_losses).mean()
        D_loss = np.array(D_losses).mean()
        epoch_lossC01 = np.array(lossC01s).mean()
        epoch_lossC02 = np.array(lossC02s).mean()
        epoch_lossC03 = np.array(lossC03s).mean()

        save_gan_train = save_gan_train.append(
            {'TotalLoss': G_loss, 'lossC01': epoch_lossC01, 'lossC02': epoch_lossC02, 'lossC03': epoch_lossC03},
            ignore_index=True)
        save_gan_train.to_csv(os.path.join(loss_dir, 'gan_train_loss_info.csv'))

        print(f'Epoch {epoch}/{num_epochs} Training g_loss : {G_loss}, d_loss : {D_loss}')

        ################################################################################################################
        # Validation
        netG.eval()

        val_G_losses = []
        val_lossC01s = []
        val_lossC02s = []
        val_lossC03s = []

        with torch.no_grad():
            val_bar = tqdm(validation_loader)
            for batch_index, batch in enumerate(tqdm(training_loader)):
                inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06 = \
                    batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                    batch[3].to(device), batch[4].to(device), batch[5].to(device)
                targetC01, targetC02, targetC03 = batch[6].to(device), batch[7].to(device), batch[8].to(device)

                outputC01, outputC02, outputC03 = netG(inputZ01, inputZ02, inputZ03, inputZ04, inputZ05, inputZ06)

                if mode == 'MSE':
                    lossC01 = mseloss(outputC01, targetC01)
                    lossC02 = mseloss(outputC02, targetC02)
                    lossC03 = mseloss(outputC03, targetC03)
                    content_loss = lossC01 + lossC02 + lossC03

                if mode == 'VGG':
                    lossC01 = vggloss(outputC01, targetC01)
                    lossC02 = vggloss(outputC02, targetC02)
                    lossC03 = vggloss(outputC03, targetC03)
                    content_loss = lossC01 + lossC02 + lossC03

                adversarial_loss = bceloss(netD(outputC01), real_label) + bceloss(netD(outputC02), real_label) + bceloss(netD(outputC03), real_label)

                val_g_loss = content_loss + 1e-3 * adversarial_loss

                val_G_losses.append(val_g_loss.item())
                val_lossC01s.append(lossC01.item())
                val_lossC02s.append(lossC02.item())
                val_lossC03s.append(lossC03.item())


                # log to tensorboard every 10 steps
                if batch_index % 1 == 0:
                    writer.add_scalar('Train/G_loss', val_g_loss.item(), epoch)
                    plot_2d_or_3d_image(targetC01, epoch * len(training_loader) + batch_index, writer, index=0, tag="Groundtruth_val/C01")
                    plot_2d_or_3d_image(targetC02, epoch * len(training_loader) + batch_index, writer, index=0, tag="Groundtruth_val/C01")
                    plot_2d_or_3d_image(targetC03, epoch * len(training_loader) + batch_index, writer, index=0, tag="Groundtruth_val/C01")
                    plot_2d_or_3d_image(outputC01, epoch * len(training_loader) + batch_index, writer, index=0, tag="Val/C01")
                    plot_2d_or_3d_image(outputC02, epoch * len(training_loader) + batch_index, writer, index=0, tag="Val/C01")
                    plot_2d_or_3d_image(outputC03, epoch * len(training_loader) + batch_index, writer, index=0, tag="Val/C01")

        val_G_loss = np.array(val_G_losses).mean()
        val_epoch_lossC01 = np.array(val_lossC01s).mean()
        val_epoch_lossC02 = np.array(val_lossC02s).mean()
        val_epoch_lossC03 = np.array(val_lossC03s).mean()

        save_gan_val = save_gan_val.append(
            {'TotalLoss': val_G_loss, 'lossC01': val_epoch_lossC01, 'lossC02': val_epoch_lossC02, 'lossC03': val_epoch_lossC03},
            ignore_index=True)
        save_gan_val.to_csv(os.path.join(loss_dir, 'gan_val_loss_info.csv'))

        print(f'Epoch {epoch}/{num_epochs} Validation g_loss : {val_G_loss}')

        # now save model parameter (from training)
        weight_g = f'G_epoch_{epoch}.pth'
        torch.save({'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    'loss': G_loss}, os.path.join(save_weight_dir, weight_g))










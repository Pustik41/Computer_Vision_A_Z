# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:15:46 2018

@author: oleksandr.pustovyi
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64
imageSize = 64 # generated image sizer (64x64)

# Creating transformation
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 

# Defining the weights_init function 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Define Generator
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                nn.Tanh()
                )
        
    def forward(self, input):
        output = self.main(input)
        return output

# Creating generator
netG = G()
netG.apply(weights_init)

# Definding the descriminator
class D(nn.Module): 
    def __init__(self): 
        super(D, self).__init__() 
        self.main = nn.Sequential( 
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            nn.Sigmoid() 
            )

    def forward(self, input): 
        output = self.main(input) 
        return output.view(-1)
   
# Creating the descrimitator    
netD = D()
netD.apply(weights_init)

# Training the DCGANs
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        # 1 step: update the weights for nn D
        netD.zero_grad()
        # Training the D with a real data
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0])) # input.size()[0] size of batch
        output = netD(input)
        errD_real = criterion(output, target)
        
        # Training the D with a fake data from the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # create noise vectors with 100 fake values and size 1x1 and with batchsize
        fake = netG(noise)
        target = Variable(torch.zeros(fake.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        # Backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        # 2 step: update the weights for nn G
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        
        #3 step: printing the losses and save the images
        print ('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %
               (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
        
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)

        
    

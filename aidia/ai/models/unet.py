import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net basic model implemented in PyTorch."""
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        
        # Encoder
        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.block1_bn = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.block2_bn = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.block3_bn = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.block4_bn = nn.BatchNorm2d(512)
        self.block4_dropout = nn.Dropout2d(0.5)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5 (Bottom)
        self.block5_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding='same')
        self.block5_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
        self.block5_bn = nn.BatchNorm2d(1024)
        self.block5_dropout = nn.Dropout2d(0.5)
        
        # Decoder
        # Block 6
        self.block6_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block6_conv1 = nn.Conv2d(1024, 512, kernel_size=2, padding='same')
        self.block6_conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding='same')
        self.block6_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.block6_bn = nn.BatchNorm2d(512)
        
        # Block 7
        self.block7_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block7_conv1 = nn.Conv2d(512, 256, kernel_size=2, padding='same')
        self.block7_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.block7_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.block7_bn = nn.BatchNorm2d(256)
        
        # Block 8
        self.block8_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block8_conv1 = nn.Conv2d(256, 128, kernel_size=2, padding='same')
        self.block8_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.block8_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.block8_bn = nn.BatchNorm2d(128)
        
        # Block 9
        self.block9_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block9_conv1 = nn.Conv2d(128, 64, kernel_size=2, padding='same')
        self.block9_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.block9_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.block9_bn = nn.BatchNorm2d(64)
        
        # Output
        self.output_conv = nn.Conv2d(64, num_classes + 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        # Block 1
        z1 = F.relu(self.block1_conv1(x))
        z1 = self.block1_conv2(z1)
        z1 = self.block1_bn(z1)
        z1 = F.relu(z1)
        z1_pool = self.block1_pool(z1)

        # Block 2
        z2 = F.relu(self.block2_conv1(z1_pool))
        z2 = self.block2_conv2(z2)
        z2 = self.block2_bn(z2)
        z2 = F.relu(z2)
        z2_pool = self.block2_pool(z2)

        # Block 3
        z3 = F.relu(self.block3_conv1(z2_pool))
        z3 = self.block3_conv2(z3)
        z3 = self.block3_bn(z3)
        z3 = F.relu(z3)
        z3_pool = self.block3_pool(z3)

        # Block 4
        z4 = F.relu(self.block4_conv1(z3_pool))
        z4 = self.block4_conv2(z4)
        z4 = self.block4_bn(z4)
        z4 = F.relu(z4)
        z4_dropout = self.block4_dropout(z4)
        z4_pool = self.block4_pool(z4_dropout)

        # Block 5 (Bottom)
        z5 = F.relu(self.block5_conv1(z4_pool))
        z5 = self.block5_conv2(z5)
        z5 = self.block5_bn(z5)
        z5 = F.relu(z5)
        z5_dropout = self.block5_dropout(z5)
        
        # Decoder
        # Block 6
        z6_up = self.block6_up(z5_dropout)
        z6 = F.relu(self.block6_conv1(z6_up))
        z6 = torch.cat([z4_dropout, z6], dim=1)
        z6 = F.relu(self.block6_conv2(z6))
        z6 = self.block6_conv3(z6)
        z6 = self.block6_bn(z6)
        z6 = F.relu(z6)
        
        # Block 7
        z7_up = self.block7_up(z6)
        z7 = F.relu(self.block7_conv1(z7_up))
        z7 = torch.cat([z3, z7], dim=1)
        z7 = F.relu(self.block7_conv2(z7))
        z7 = self.block7_conv3(z7)
        z7 = self.block7_bn(z7)
        z7 = F.relu(z7)
        
        # Block 8
        z8_up = self.block8_up(z7)
        z8 = F.relu(self.block8_conv1(z8_up))
        z8 = torch.cat([z2, z8], dim=1)
        z8 = F.relu(self.block8_conv2(z8))
        z8 = self.block8_conv3(z8)
        z8 = self.block8_bn(z8)
        z8 = F.relu(z8)
        
        # Block 9
        z9_up = self.block9_up(z8)
        z9 = F.relu(self.block9_conv1(z9_up))
        z9 = torch.cat([z1, z9], dim=1)
        z9 = F.relu(self.block9_conv2(z9))
        z9 = self.block9_conv3(z9)
        z9 = self.block9_bn(z9)
        z9 = F.relu(z9)
        
        # Output
        output = torch.sigmoid(self.output_conv(z9))
        
        return output

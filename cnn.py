import torch
import troch.nn as nn

class my_cnn(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(my_cnn, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.feature = nn.Sequential(
#1                
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=24,
                #kernel_size=(param[para_loop][0],param[para_loop][1]),
                kernel_size=(9,7),
                stride=1,
                padding = (0,0),
            ),
            nn.ReLU(),                                            
            nn.MaxPool2d(kernel_size=(2,2)),            
#2            
            nn.Conv2d(
                in_channels=24,
                out_channels=28,
                kernel_size=(12,8),
                stride=1,
                padding = (0,0),
            ),  
            nn.ReLU(),                  
            nn.MaxPool2d(kernel_size=(4,4)),            

#3            
            nn.Conv2d(
                in_channels=28,
                out_channels=32,
                kernel_size=(6,4),
                stride=1,
                padding = (0,0),
            ),  
            nn.ReLU(),                                
            # nn.MaxPool2d(kernel_size=(2,2)),            
        )

        self.classification = nn.Sequential(
            # nn.Dropout(p=0.5),
            #nn.Linear(in_features=16 * param[para_loop][2] * param[para_loop][3], out_features=64),
            nn.Linear(in_features=32 * 3 * 2, out_features=32),            # nn.Dropout(p=0.5),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=32 , out_features=n_classes),       
        )
       
    def forward(self, x):
        x = self.feature(x)
        temp = x.view(x.shape[0], -1)
        output = self.classification(temp)
        return output, x

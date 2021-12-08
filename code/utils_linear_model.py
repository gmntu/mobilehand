###############################################################################
### Adapted from PyTorch implementation of HMR
###     MandyMo: https://github.com/MandyMo/pytorch_HMR/blob/master/src/LinearModel.py
###############################################################################

import sys
import torch
import numpy as np
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(LinearModel, self).__init__()
        # Input param
        self.fc_layers   = fc_layers   # List of int of neuron layer: [2048+26, 1024, 1024, 26]
        self.use_dropout = use_dropout # List of bool to use dropout: [True, True, False]
        self.drop_prob   = drop_prob   # List of float for drop prob: [0.5, 0.5, 0]
        self.use_ac_func = use_ac_func # List of bool to use active function: [True, True, False]
        
        if not self._check():
            msg = '[LinearModel] Wrong parameters!'
            print(msg)
            sys.exit(msg)

        self.create_layers()


    def _check(self):
        while True:
            if not isinstance(self.fc_layers, list):
                print('fc_layers require list, get {}'.format(type(self.fc_layers)))
                break
            
            if not isinstance(self.use_dropout, list):
                print('use_dropout require list, get {}'.format(type(self.use_dropout)))
                break

            if not isinstance(self.drop_prob, list):
                print('drop_prob require list, get {}'.format(type(self.drop_prob)))
                break

            if not isinstance(self.use_ac_func, list):
                print('use_ac_func require list, get {}'.format(type(self.use_ac_func)))
                break
            
            l_fc_layer = len(self.fc_layers)
            l_use_drop = len(self.use_dropout)
            l_drop_prob = len(self.drop_prob)
            l_use_ac_func = len(self.use_ac_func)

            return l_fc_layer >= 2 and \
                    l_use_drop < l_fc_layer and \
                    l_drop_prob < l_fc_layer and \
                    l_use_ac_func < l_fc_layer and \
                    l_drop_prob == l_use_drop

        return False


    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_prob = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)

        self.fc_blocks = nn.Sequential()
        
        for _ in range(l_fc_layer - 1):
            self.fc_blocks.add_module(
                name = 'regressor_fc_{}'.format(_),
                module = nn.Linear(in_features=self.fc_layers[_], out_features=self.fc_layers[_+1])
            )
            
            if _ < l_use_ac_func and self.use_ac_func[_]:
                self.fc_blocks.add_module(
                    name = 'regressor_af_{}'.format(_),
                    module = nn.ReLU()
                )
            
            if _ < l_use_drop and self.use_dropout[_]:
                self.fc_blocks.add_module(
                    name = 'regressor_fc_dropout_{}'.format(_),
                    module = nn.Dropout(p=self.drop_prob[_])
                )


    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        sys.exit(msg)


###############################################################################
### Simple example to test the program                                      ###
###############################################################################
if __name__ == '__main__':
    # Just to check network architecture
    device      = torch.device('cpu')
    fc_layers   = [576+39, 288, 288, 39]
    use_dropout = [True, True, False]
    drop_prob   = [0.5, 0.5, 0]
    use_ac_func = [True, True, False]
    model = LinearModel(fc_layers, use_dropout, drop_prob, use_ac_func).to(device)
    print(model)
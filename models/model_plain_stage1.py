from models.model_plain import ModelPlain
import numpy as np


class ModelPlainStage1(ModelPlain):
    """Train with four inputs (L, L1, L2, H) and with pixel loss"""

    # ----------------------------------------
    # feed L/LT/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.LT = data['LT'].to(self.device)   

    # ----------------------------------------
    # feed L to netE
    # ----------------------------------------
    def netE_forward(self):
        self.E = self.netE(self.L)
    
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

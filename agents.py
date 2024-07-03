import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter, get_common_distributions
import os
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# from PIL import Image


from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr
from collections import deque
from scipy.stats import genextreme

class Household(Agent):
    def __init__(self, unique_id: int, model: Model, x, y,
                 fgrote, fmiddel, fklein, feklein,
                 r_cap, income, seniority,expenditure, fund, ltv, install,
                 v, sp, tm):
        super().__init__(unique_id, model)
        # location
        self.x = x
        self.y = y

        # flood param
        self.fgrote= fgrote
        self.fmiddel = fmiddel
        self.fklein = fklein
        self.feklein = feklein

        self.f = 0
        self.f_d = 0

        self.u = 1.0 # neighborhood utility

        # 2 status param
        self.s_e = 1
        self.s_d = 0

        self.income = income
        self.sen = seniority
        self.expend = expenditure
        self.fund = fund
        self.tue = 0
        self.ltv = ltv
        self.install = install
        self.tm = tm

        self.r_cap = r_cap
        self.r_inst = self.install / self.income
        
        self.v = v
        self.c = v
        # self.c_for_util = v
        self.sp = sp
        self.v_arr = deque(np.zeros(36))

        self.score = 210

    def update_s_e(self):
        """update employment status"""
        rv = np.random.uniform(0, 1)
        r_e = self.model.r_e
        if self.s_e == 1 and rv < r_e:
            self.s_e = 0
        if self.s_e == 0 and rv >= r_e:
            self.s_e =1

    def sigmoid(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # Use the more numerically stable formula for negative x
            return np.exp(x) / (1 + np.exp(x))
    
    def update_s_d(self):
        """update default status"""
        share_income = self.share_income()
        # hard margin
        if self.s_d == 0 and self.fund - max(share_income * self.income, 500) <= self.install:
            self.s_d = 1

        # soft margin
        self.u = self.neighbor_utility()
        x = (self.install * self.tm + self.model.alpha) / (self.income + self.model.alpha) * (self.model.beta3*(1-self.s_e)*self.tue + 1)
        p_wtp = self.model.beta1 * self.u -self.model.beta2 * self.sigmoid(x)
        # p_wtp = self.model.beta1 * u
        rv = np.abs(np.random.normal(0, self.model.std_stp))
        # rv = 0.05
        if self.s_d == 0 and p_wtp < rv:
            self.s_d = 1
        elif self.s_d ==1 and p_wtp > (1-rv):
            self.s_d = 0

    def new_income(self):
        """update income"""
        old_s_e = self.s_e
        self.update_s_e()
        new_s_e = self.s_e
        # in the update Se function, the status can only be moved with 1 step
        # it's not possible to let the agent to have diff income from diff comopany in the sequential month
        if old_s_e ==1 and new_s_e==1: 
            pass
        elif new_s_e==0 and old_s_e==1:
            self.income = max(0.7 * self.income, 5000)
        elif new_s_e==1 and old_s_e==1 and self.tue>3:
            self.income = 0
        elif new_s_e==1 and old_s_e==0:
            self.income = burr.rvs(c=3.30, d=0.45, loc=-12.76, scale=3101.46)
            # distribution params can be saved in config file

    def neighbor_utility(self):
        """compute neighborhood utility"""
        #  sum c/ sum v
        tot_c = 0
        tot_v = 0
        neighbors_grid = self.model.grid.get_neighbors((self.x, self.y), moore=True, radius=self.model.grid_radius, include_center=True)
        for neighbor in neighbors_grid:
            tot_c += neighbor.c
            tot_v += neighbor.v
        
        u = tot_c / tot_v
        # return normalized utility
        if u < 0:
            return 0
        elif u >= 1:
            return 1
        else:
            return (u - self.model.u_min) / (self.model.u_max - self.model.u_min)
            # return u


    def share_income(self):
        """return the share of income, in order to compute expenditure of the agents"""
        return 2.7 / (1 + 0.85 * self.income) + 0.3
    
    def compute_score(self):
        """compute pd score"""
        score = 0
        # ltv
        if self.ltv <= 0.4:
            score += 20
        elif self.ltv <=0.7:
            score += 14
        elif self.ltv <= 0.9:
            score += 7

        # income
        if self.income <= 300:
            score += 12
        elif self.income <= 500:
            score += 17
        elif self.income <=800:
            score += 24
        elif self.income <= 1700:
            score += 34
        else:
            score += 38

        # seniority
        if self.sen <= 15:
            score += 9
        elif self.sen <= 47:
            score += 14
        else:
            score += 28

        # r_cap
        if self.r_cap <=0.05:
            score += 9
        elif self.r_cap <= 0.40:
            score += 22
        elif self.r_cap <= 0.50:
            score += 24
        else:
            score += 33
        
        # r_inst
        if self.r_inst <= 0.40:
            score += 20
        else:
            score += 9
        
        # credit score
        ratio_arr = np.sum(self.v_arr) / 36
        score += 100 * (1 - ratio_arr)

        # update score
        self.score = score
    
    def find_flood_map(self):
        """find the respective flood risk map data for agents"""
        gev = self.model.gev
        # flood impact
        # given a GEV number --> return period number --> how severe the flood is 
        if 0 < gev <= 30:
            # grote kans overstroming
            self.f_d = self.fgrote
        elif 30< gev <= 300:
            # mid-grote kans
            self.f_d = self.fmiddel
        elif 300 < gev <=3000:
            # klein kans
            self.f_d = self.fklein
        elif 3000 < gev:
            # zeer klein kans
            self.f_d = self.feklein
        elif gev <= 0:
            self.f_d = 0
        # make gev and f_r(t) global
        
        # find f_r
        # f_r = self.model.f_r

        # if gev > 0: # if second round flood happens
        #     self.f = 0.7 * self.f + 0.3 * self.f_d
        #     # self.f += self.f_d
        # else:
        #     self.f = self.f * 0.8
        #     # self.f = self.f_d * ( 1 - f_r )
        self.f = 0.3 * self.f + 0.5 * self.f_d
    
    def update(self):
        # with the given s_e and s_d, update other variables & also update state variables
        #  note: installments stays a constant and no not need to be updated

        # find f_d, f
        self.find_flood_map()
        # update collateral
        delta_price =  self.sp * self.f * 717
        self.c = self.v - delta_price
        # self.c_for_util = self.v - self.model.beta4 * delta_price

        # seniority
        self.sen = self.s_e * (self.sen + 1) + (1 - self.s_e) * self.sen
        # duration unemployed
        self.tue = self.s_e * 0 + (1 - self.s_e) * (self.tue + 1)

        # decide default status and decide the respective expenditure
        # expenditure
        share_income = self.share_income()
        self.update_s_d()
        self.expend = max(share_income * self.income - self.install * (1 - self.s_d) , 500) + (1 - self.s_d) * self.install
        self.tm -= (1-self.s_d)

        self.v_arr.popleft()
        self.v_arr.append(self.s_d)
        
        self.fund = self.fund + self.income - self.expend

        # update income, update Se is included in the new_income function
        self.new_income()

        # update score & terms in the score components
        self.ltv = self.install * self.tm / self.c
        if self.tm != 0:
            self.r_cap = self.fund / (self.tm * self.install)
        # avoid zero division
        self.r_inst = self.install / (self.income+0.01)
        self.compute_score()

    def step(self):
        self.update()

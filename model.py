import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter, get_common_distributions
import os
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr
from collections import deque
from scipy.stats import genextreme
import joblib

from agents import Household
from utils import density_preprocess, find_coordi, calculate_emi, scorecard
# from config import interest_rate, n_new_joiner, climate_risk
import random

random.seed(42)

# preparation folder
FOLDER = 'preparation'
# load kernel
paths = ['kde_model_3d.joblib', 'kde_model_nvm.joblib']
ing_kde = joblib.load(FOLDER + '/' + paths[0])
nvm_kde = joblib.load(FOLDER + '/' + paths[1])

class ESGMotgageModel(Model):
    def __init__(
            self, 
            width, 
            height, 
            density_map, 
            m_fgrote, 
            m_fmiddel, 
            m_fklein, 
            m_feklein,
            binary_map,
            grid_radius,
            acpt_score,
            gev_list):
        
        self.add_agent_controller = True
        self.remove_thresh = 12

        self.acpt_score = acpt_score
        
        self.num_agents = 1 # max density per unit
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.tot_num_agents = 0
        self.tot_num_agents_his = 0
        self.num_default = 0
        self.num_mature = 0
        self.num_new_join = 0
        self.num_step = 0

        # save map
        self.density_map = density_map
        self.m_fgrote = m_fgrote
        self.m_fmiddel = m_fmiddel
        self.m_fklein = m_fklein
        self.m_feklein = m_feklein
        self.binary_map = binary_map

        # gev
        self.gev_list = gev_list

        # global variables
        self.grid_radius = grid_radius

        self.gev = 0

        self.beta = 0.1

        self.u_min = 0.50
        self.u_max = 1.0

        self.beta1 = 1.00
        self.beta2 = 0.20
        self.beta3 = 1.50
        # self.beta4 = 1000
        self.alpha = 1
        self.std_stp = 0.5 # std for random number to judgw the willingness to pay
        self.r_e = 0.1

        # new joiners
        self.gamma = 1 / 25
        self.epsilon = 1.0
        self.std_nj = 5.0

        self.mu_i = 0.04
        self.std_i = 0.001

        self.gu_min = 0.90
        self.gu_max = 1.0

        self.N_nj = 500

        # Create agents based on probability density map
        self.populate_agents() 
        
        self.failures = pd.DataFrame()
        self.matures = pd.DataFrame()

        # set the datacollector
        model_reporters = {"gev": 'gev',
                           'epsilon': 'epsilon',
                           'r_e': 'r_e',
                           'tot_num_agents': 'tot_num_agents',
                           'tot_num_agents_his': 'tot_num_agents_his',
                           'num_default': 'num_default',
                           'num_mature': 'num_mature',
                           'num_new_join':'num_new_join'}
        
        agent_reporters = {'score': 'score',
                           's_d': 's_d',
                           's_e': 's_e',
                           'x':'x',
                           'y': 'y',
                           'u':'u',
                           'income': 'income',
                           'expend': 'expend',
                           'fund': 'fund',
                           'seniority': 'sen',
                           'ltv': 'ltv',
                           'install': 'install',
                           'sp': 'sp',
                           'tm': 'tm',
                           'r_cap':'r_cap',
                           'r_inst': 'r_inst',
                           'v':'v',
                           'c':'c',
                           'v_arr':'v_arr'
                           }
        # r_cap, income, seniority,expenditure, fund, ltv, install, v, sp, tm
        self.datacollector = DataCollector(
            model_reporters=model_reporters, 
            agent_reporters=agent_reporters
            )

        

    def init_agent(self, x, y, new_join = False):
        # flood risk map values
        fgrote = self.m_fgrote[x,y]
        fmiddel = self.m_fmiddel[x,y]
        fklein = self.m_fklein[x,y]
        feklein = self.m_feklein[x,y]

        # seniority = np.floor(np.random.beta(a=0.88, b=2.79) * 606.34 + 1.0)

        ltv = np.abs(np.random.normal(0.57, 0.1))
        # sp = burr.rvs(c=2.20, d=3.00, loc=-0.63, scale=146.68)
        # r_cap = burr.rvs(c=9.42, d=0.14, loc=-0.11, scale=73.40)
        


        if new_join == False:
            # income, job_since, ratio cap respectively
            income, seniority, r_cap = ing_kde.sample(1)[0]
            v, sp = nvm_kde.sample(1)[0]
            v *= 1000 # from keuro to euro
            # income = burr.rvs(c=3.30, d=0.45, loc=-12.76, scale=3101.46)
            # expenditure is computed from the share of the income
            share_income = 2.7/(1+0.85*income)+0.3
            expenditure = share_income * income
            # v = burr.rvs(c=3.38, d=0.92, loc=-1.34, scale=195.25) * 1000 # unit: euro
            tm = np.random.uniform(1, 120)
            install = ltv * v / tm
        else:        
            # while True:
            #     income, seniority, r_cap = ing_kde.sample(1)[0]
            #     # income = burr.rvs(c=3.30, d=0.45, loc=-12.76, scale=3101.46)
            #     if income >= 1000:
            #         break
            income, seniority, r_cap = ing_kde.sample(1)[0]
            # expenditure is computed from the share of the income
            share_income = 2.7/(1+0.85*income)+0.3
            expenditure = share_income * income
            tm = 120
            mortgage_amount = 4 * 12 * income
            ri = np.random.normal(self.mu_i, self.std_i) # annually interest rate
            install = calculate_emi(mortgage_amount, ri, loan_tenure_years = 10)
            # coeff_ri = ri * (1+ri)**480 / ((1+ri)**480 - 1)
            # install = mortgage_amount * coeff_ri
            v = mortgage_amount / ltv
            sp = burr.rvs(c=2.20, d=3.00, loc=-0.63, scale=146.68)
            
        
        fund = r_cap * tm * install
        
        return fgrote, fmiddel, fklein, feklein, r_cap, income, seniority,expenditure, fund, ltv, install, v, sp, tm



    def populate_agents(self):
        """initialize agents, add them in the system"""
        unique_id = 0
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                num_agents_at_location = int(round(self.density_map[x, y] * self.num_agents))

                for _ in range(num_agents_at_location):
                    # # add flood risk
                    # fgrote = self.m_fgrote[x,y]
                    # fmiddel = self.m_fmiddel[x,y]
                    # fklein = self.m_fklein[x,y]
                    # feklein = self.m_feklein[x,y]

                    # income = burr.rvs(c=3.30, d=0.45, loc=-12.76, scale=3101.46)
                    # seniority = np.random.beta(a=0.88, b=2.79) * 606.34 + 1.0
                    # # expenditure is computed from the share of the income
                    # share_income = 2.7/(1+0.85*income)+0.3
                    # expenditure = share_income * income
                    # # fund = 
                    # ltv = np.random.normal(0.57, np.sqrt(10))
                    # # install =
                    # v = burr.rvs(c=3.38, d=0.92, loc=-1.34, scale=195.25) * 1000 # unit: euro
                    # sp = burr.rvs(c=2.20, d=3.00, loc=-0.63, scale=146.68)
                    # tm = np.random.uniform(0, 480)
                    # r_cap = burr.rvs(c=9.42, d=0.14, loc=-0.11, scale=73.40)
                    
                    # install = ltv * v / tm
                    # fund = r_cap * tm * install

                    # # add agent
                    # agent = Household(unique_id, self, x, y,
                    #                     fgrote, fmiddel, fklein, feklein,
                    #                     r_cap, income, seniority,expenditure, fund, ltv, install,
                    #                     v, sp, tm)
                    args = self.init_agent(x, y, new_join = False)
                    agent = Household(unique_id, self, x, y, *args)
                    self.schedule.add(agent)
                    self.grid.place_agent(agent, (x, y))

                    unique_id += 1
                    self.tot_num_agents += 1
                    self.tot_num_agents_his += 1

    def global_utility(self):
        """influence the number of new joiners"""
        tot_c = np.sum([agent.c for agent in self.schedule.agents])
        tot_v = np.sum([agent.v for agent in self.schedule.agents])
        utility = tot_c / tot_v
        # normalization global utility
        epsilon =  (utility - self.gu_min) / (self.gu_max - self.gu_min)
        if epsilon <0:
            return 0
        else:
            return epsilon
        # return utility
    
    def find_position(self, size):
        """find initial location for new joiners of size=n."""
        density_slots, coordinates = density_preprocess(self.density_map)
        # print(np.shape(self.density_map))
        rvs = np.random.uniform(0, 1, size=size)
        y_list = []
        x_list = []
        for rv in rvs:
            y, x = find_coordi(rv, density_slots, coordinates)
            # print(y, x)
            x_list.append(x)
            y_list.append(y)
        # for _ in range(size):
        # # while True:
        #     y = np.random.uniform(0, self.binary_map.shape[0])
        #     x = np.random.uniform(0, self.binary_map.shape[1])
        #     if self.binary_map[int(y), int(x)] == 1:
        #         y_list.append(int(y))
        #         x_list.append(int(x))
        #         counter += 1
        #         # if counter == size:
        #         #     break

        return y_list, x_list

    def add_agent(self):
        """add new agents & initialize them"""
        self.epsilon = self.global_utility()
        # coef - gamma 1/100 (according to the uniform distribution with increasing trend), as tm is uniform distribution
        num_nj = np.floor(self.gamma * self.epsilon* self.N_nj) + np.floor(np.random.normal(0, self.std_nj))
        
        if num_nj<=0:
            num_nj = 0
            return
        else:
            num_nj = int(num_nj)

        print(f'num_new={num_nj}')

        x_list, y_list = self.find_position(num_nj)
        unique_id = self.tot_num_agents_his
        for i in range(num_nj):
            x, y = x_list[i], y_list[i]
            # while True:
            # initializa new joiners, add them to the system
            args = self.init_agent(x, y, new_join = True)
            # acceptance rule
            score = scorecard(*args[4:])
            # if score > self.acpt_score:
            #     break
            if score >= self.acpt_score:
                agent = Household(unique_id, self, x, y, *args)
                self.schedule.add(agent)
                self.grid.place_agent(agent, (x, y))

                unique_id += 1
                self.num_new_join += 1
                self.tot_num_agents += 1
                self.tot_num_agents_his += 1
                print(f'house price: {agent.v}, income:{agent.income}, install:{agent.install}, expenditure:{agent.expend}')


    def remove_agent(self):
        """remove matured/ defaulted agents"""
        # self.schedule.remove(agent)
        # self.grid.remove_agent(agent)

        for agent in self.schedule.agents:        
            # remove dead agents
            if np.sum(agent.v_arr) >= self.remove_thresh:
                self.tot_num_agents -= 1
                self.num_default += 1
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
                agent_info = {'step': self.num_step,
                              'id': agent.unique_id,
                              'score': agent.score,
                           'x':agent.x,
                           'y': agent.y,
                           'u':agent.u,
                           'income': agent.income,
                           'expend': agent.expend,
                           'fund': agent.fund,
                           'seniority': agent.sen,
                           'ltv': agent.ltv,
                           'install':agent.install,
                           'sp': agent.sp,
                           'tm': agent.tm,
                           'r_cap':agent.r_cap,
                           'r_inst': agent.r_inst,
                           'v': agent.v,
                           'c': agent.c,
                           'v_arr':agent.v_arr
                           }
                self.failures = self.failures._append(agent_info, ignore_index = True)
            # remove matured agents
            elif np.floor(agent.tm) <= 0:
                self.tot_num_agents -= 1
                self.num_mature += 1
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
                agent_info = {'step': self.num_step,
                              'id': agent.unique_id,
                              'score': agent.score,
                           'x':agent.x,
                           'y': agent.y,
                           'u':agent.u,
                           'income': agent.income,
                           'expend': agent.expend,
                           'fund': agent.fund,
                           'seniority': agent.sen,
                           'ltv': agent.ltv,
                           'install':agent.install,
                           'sp': agent.sp,
                           'tm': agent.tm,
                           'r_cap':agent.r_cap,
                           'r_inst': agent.r_inst,
                           'v': agent.v,
                           'c': agent.c,
                           'v_arr':agent.v_arr
                           }
                self.matures = self.matures._append(agent_info, ignore_index = True)


    # def gev_flood_occur(self):
    #     """generate gev number to control flood occurence"""
        # gev_dist = genextreme(c=-self.k, loc=self.b, scale=self.a)
        # return  gev_dist.rvs(size=1)[0]
        # return self.gev_list[self.num_step]

    
    def aggregate_score(self):
        """return statistics of agent scores"""
        scores = [agent.score for agent in self.schedule.agents]
        mean = np.mean(scores)
        std = np.std(scores)
        # TODO: add other statistics
        return mean, std

    def sigmoid(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # Use the more numerically stable formula for negative x
            return np.exp(x) / (1 + np.exp(x))
    

    def step(self):
        # generate GEV flood
        # self.gev = self.gev_flood_occur()
        self.gev = self.gev_list[self.num_step]
        # update employment rate, income shock
        if self.gev >0:
            self.r_e = 0.1 + np.tanh(self.gev / 100) * self.beta
        else:
            self.r_e = 0.1
        print(f'rate of unemployment: {self.r_e}')


        # bank aggregate client data
        mean, std = self.aggregate_score()

        # self.schedule.step()
        if self.add_agent_controller == True:
            print('add new agent')
            self.add_agent()

        self.remove_agent()
        
        
        self.num_step += 1
        
        # add data collector
        # all_v = [agent.v for agent in self.schedule.agents]
        # print(f"avg housing price: {np.mean(all_v)}")
        print(f'epoch = {self.num_step}')
        # employed = [agent.s_e for agent in self.schedule.agents]
        # defaulted = [agent.s_d for agent in self.schedule.agents]
        # utility = [agent.u for agent in self.schedule.agents]
        # print(f'num_employed: {np.sum(employed)}, num_s_d = 1: {np.sum(defaulted)}')
        # print(f'avg utility = {np.mean(utility)}, min {np.min(utility)}, max {np.max(utility)}')
        # print(utility[:20])
        # print(f'score card: mean = {mean}, std = {std}.')
        # scores = [agent.score for agent in self.schedule.agents]
        # print(f'scores: Q1={np.percentile(scores, 25)}, Q2={np.percentile(scores, 50)}, Q3={np.percentile(scores, 75)}')
        print(f"tot_a={self.tot_num_agents}, tot_his={self.tot_num_agents_his}, num_default={self.num_default}, num_mature={self.num_mature}, num_new={self.num_new_join}")
        # print(self.gev, self.epsilon)

        self.datacollector.collect(self)
        self.schedule.step()
        

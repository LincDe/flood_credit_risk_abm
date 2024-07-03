import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter, get_common_distributions
import os
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image


from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr
from collections import deque
from scipy.stats import genextreme

from agents import Household
from model import ESGMotgageModel
import argparse
from pathlib import Path
from config import b_set, a_set, k_set
import random

random.seed(42)

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
# read maps
FOLDER = ROOT_DIR + '/' + 'preparation'
paths = ['grote.png', 'middel.png', 'kleine.png', 'extreme_kleine.png', 'nvm_cleaned.png']

def read_map(FOLDER, path):
    path = FOLDER + '/' + path
    map = Image.open(path)
    map = np.array(map.convert("L"))
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            map[i][j] = 255-map[i][j]
    # normalization
    return map / np.max(map)


m_fgrote = read_map(FOLDER, paths[0])
m_fmiddel = read_map(FOLDER, paths[1])
m_fklein = read_map(FOLDER, paths[2])
m_feklein = read_map(FOLDER, paths[3])
nvm = read_map(FOLDER, paths[4])

# create a binary map to restrict location of new joiner
binary_map = np.zeros(nvm.shape)
print(binary_map.shape)
for i in range(nvm.shape[0]):
    for j in range(nvm.shape[1]):
        if nvm[i, j] != 0:
            binary_map[i, j] = 1


# Example usage
width = nvm.shape[0]
height = nvm.shape[1]

print(f'width: {width}, height: {height}')

# GEV_df = pd.read_csv(FOLDER+ '/' + 'gev_config.csv')

def main(tot_step, grid_radius, acpt_score, folder, settings, repeats, add_agents):
    
    if add_agents == True:
        # this file contains 1500 numbers/setting
        GEV_df = pd.read_csv(FOLDER+ '/' + 'gev_config3.csv')
        # this file contains 1000 numbers/setting
        # GEV_df = pd.read_csv(FOLDER+ '/' + 'gev_config.csv')
    else:
        GEV_df = pd.read_csv(FOLDER+ '/' + 'gev_config2.csv')
        
    # ORIGINAL:
    res_gev_list = GEV_df['set'+str(settings)]

    # for KC with different GEV seeds
    # if settings == 0:
    #     res_gev_list = GEV_df['set'+str(settings)]
    # else:
    #     res_gev_list = GEV_df['set'+str(settings)+'_'+str(count)]
    #     print('Scenario: set'+str(settings)+'_'+str(count))



    # # # scenario1: base1-300, 400-500, stress 300-400
    # set1 = GEV_df['set1']
    # res_gev_list = list(set1[0:600]) + list(gev_list[601:1400])
    # res_gev_list = list(set0[0:300]) + list(gev_list[301:400]) + list(set0[401: 600])

    # # res_gev_list = list(set0[0:300]) + list(gev_list[301:700]) 

    # Create and run the model
    model = ESGMotgageModel(width, height, nvm, m_fgrote, m_fmiddel, m_fklein, m_feklein, binary_map , grid_radius, acpt_score, res_gev_list)
    
    model.add_agent_controller = add_agents
    print(f'add new agents: {model.add_agent_controller}')
    # update GEV param settings according to --settings 
    # model.b = b_set[settings]
    # model.a = a_set[settings]
    # model.k = k_set[settings]
   
    model.num_agents = 1.2
    print(f'setting {settings}')
    # print(f'GEV params: b={model.b}, a = {model.a}, k = {model.k}')
    print(f'total number of agents: {model.tot_num_agents}')

    # Run the model for a certain number of steps
    for _ in range(tot_step):
        # for MV process, break if all agents are removed (without adding agents to the system)
        if model.tot_num_agents <= 0:
            break
        model.step()
        
    # save the resultes
    save_path = ROOT_DIR + '/' +  folder
    if Path(save_path).is_dir()==False:
        os.mkdir(save_path)
    model_output = model.datacollector.get_model_vars_dataframe()
    model_output.to_csv(save_path +f'/model_output_score{acpt_score}_r{grid_radius}_set{settings}_{count}.csv')
    agents_output = model.datacollector.get_agent_vars_dataframe()
    agents_output.to_csv(save_path +f'/agents_output_score{acpt_score}_r{grid_radius}_set{settings}_{count}.csv')

    # dict1 = {'default_score': model.failures}
    # df = pd.DataFrame(dict1)
    model.failures.to_csv(save_path +f'/default_score{acpt_score}_r{grid_radius}_set{settings}_{count}.csv')

    # dict2 = {'mature_score': model.matures}
    # df = pd.DataFrame(dict2)
    model.matures.to_csv(save_path+f'/mature_score{acpt_score}_r{grid_radius}_set{settings}_{count}.csv')


# Visualize the agent locations
# # plt.figure(figsize=(10,20))
# agent_positions = [(agent.x, agent.y) for agent in model.schedule.agents]
# plt.imshow(nvm, cmap='Blues', interpolation='none')
# w, h = zip(*agent_positions)
# plt.scatter(h, w, color='red', marker='.', label='Households')
# plt.legend()
# plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tot_step', default=1, type=int,
                        help='simulation time step')
    parser.add_argument('--grid_radius', default=30, type=int,
                        help='radius of neighborhood')
    parser.add_argument('--acpt_score', default=205.7, type=float,
                        help='Cut-off threshold of accepting clients')
    parser.add_argument('--folder', default=f'out1/test', type=str,
                        help='folder to save the set of outcome files with differenc experimental setting')
    parser.add_argument('--settings', default=1, type=int,
                        help='GEV parameter settings (numbering starts from 1), 0 means no flood')
    parser.add_argument('--repeats', default=5, type=int,
                        help='Repeat the same settings for ... times.')
    parser.add_argument('--add_agents', action='store_false',
                        help='Default: add new agents, with this action, can test without adding new agents')
    # parser.add_argument('--no_flood', action='store_true',
    #                     help='Default: create flood events. in conparison, put this no flood situation where completely no flood happens. ')

    args = parser.parse_args()
    kwargs = vars(args)
    # kargs returns a dict:{var name: var value}
    # print(kwargs['repeats'])

    for count in range(kwargs['repeats']):
        main(**kwargs)

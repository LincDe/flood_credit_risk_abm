# Master thesis project: ESG climate risk stress testing with flood as a key risk factor in mortgage models: an agent-based approach

This project provide a agent-based model that studies the recurrent flood impact on the mortgage portfolios.

## data preparation: 

The flood risk map can be found: [Flood|Risk map][https://www.risicokaart.nl/kaarten/risicosituaties/overstroming]  
The initialization of the agents is based on the distribution extracted by KDE method with private data, the features per agent are random sampled from this distribution.

## File structure:

agents.py - contains the mesa agent class
model.py - contains the mesa model class
run.py - the function to excute running. A separated .job or .sh file if the simulation is supposed to be run in a visual machine.
utils.py - contains helper functions 
config.py - contains the GEV parameters to generate recurrent flood with different intensity

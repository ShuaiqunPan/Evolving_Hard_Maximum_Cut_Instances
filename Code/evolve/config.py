import os

'''
The ratio of GW/RQAOA; In practice, we run the CMA-ES with minimization. So the generated instances
with this ratio which smaller than 1 denotes the instances achieve better performance on RQAOA than GW.
'''

output_figures_directory = "output_figures_GW_RQAOA/"
output_others_directory = "output_others_GW_RQAOA/"
output_CMAES_directory = "output_CMAES_GW_RQAOA/"

os.makedirs(output_figures_directory, exist_ok=True)
os.makedirs(output_others_directory, exist_ok=True)
os.makedirs(output_CMAES_directory, exist_ok=True)

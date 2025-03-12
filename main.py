##########################################################################
# Copyright 2025 Cai Jianping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
# This script is used to perform experiments for FL-DPCR
##########################################################################
from fldpcr.server import Server
import warnings

warnings.filterwarnings("ignore")

def bestK(t, func):
    mi = 2 ** 50
    res = -1
    for k in range(40):
        r = func(k);
        if abs(r - t) < mi:
            mi = abs(r - t)
            res = k
    return res


def runFLDPCR(configs):
    ## Configuration Information
    global_config = configs["global_config"]
    data_config = configs["data_config"]
    fed_config = configs["fed_config"]
    optim_config = configs["optim_config"]
    init_config = configs["init_config"]
    model_config = configs["model_config"]
    dp_config = configs["dp_config"]
    dpcr_model = configs["dpcr_model"]

    message = "\n[WELCOME] Unfolding configurations...!"
    print(message);
    for key in configs:
        print({key: configs[key]});

    if dpcr_model['name'].startswith('kTCR'):
        from dpcrpy import gen
        gen(**dpcr_model).preSolvePrivateBudget()

    ## Start Run
    # initialize federated learning
    central_server = Server(model_config, global_config, data_config, init_config, fed_config, optim_config, dp_config,
                            dpcr_model)
    central_server.setup()

    # do federated learning
    central_server.fit()

    message = "...done all learning process!\n...exit program!"
    print(message)


if __name__ == '__main__':
    # Set experimental parameters
    config = {
        'global_config': {
            # If usingDP = True, private federated learning is achieved using differential privacy.
            # The communication round number is indicated by fed_config.R
            # Otherwise, the federated learning is without privacy.
            'usingDP': True,
            'device': 'cuda',  # GPU or CPU
        },
        'data_config': {  # dataset infomation.
            'data_path': 'data',
            'dataset_name': 'CIFAR10',  # Available Datasets: MNIST, FashionMNIST, CIFAR10
            'num_shards': 200,
        },
        'fed_config': {
            'C': 1,  # Percentage of participants participating in training per communication round
            'K': 20,  # Number of participants
            # Maximum communication rounds. It may be less than actual communication rounds due to exhaustion of privacy budget.
            'R': 50,
            'E': 100,  # Internal iteration number
            'sample_rate': 0.01,  # Sampling rate of each iteration
        },
        'optim_config': {
            'lr': 0.1  # Learning rate
        },
        'init_config': {
            'init_type': 'xavier',
            'init_gain': 1.0,
            'gpu_ids': [0]
        },
        'model_config': {
            # The available models see the file models/__init__.py. If None, then adopt the NN model recommended by the dataset.
            'name': None,
        },
        'dp_config': {
            # Privacy Parameters for (epsilon, delta)-DP
            'epsilon': 8,
            'delta': 4.0e-4,
            'max_norm': 1.0,
            # If 'isFixedClientT' is True, the iteration number of participants is indicated by using 'ClientT' and sigma is calculated by the private budget and 'ClientT'.
            # Otherwise, use 'sigma' to indicate the amount of noise added by the participant, and 'ClientT' is calculated based on the privacy budget and 'sigma'
            'isFixedClientT': True,
            'clientT': 5000,  # Only works for isFixedClientT = True
            'sigma': 1  # Only works for isFixedClientT = False
        },
        'dpcr_model': {
            # Available Compared DPCR models: DPFedAvg (without DPCR), SimpleMech, TwoLevel, BinMech, FDA, BCRG, ABCRG, Honaker
            # Our DPCR models:
            #   kTCR_k2, kTCR model with k==2
            #   kTCR_k3, kTCR model with k==3
            #   kTCR_k5, kTCR model with k==5
            #   kTCR_k8, kTCR model with k==8
            #   kTCR_k10, kTCR model with k==10
            'name': 'kTCR_k2',
            'args': {}  # Automatic setting by clientT and the DPCR model
        },
    }
    # Refine the parameter settings.
    clientT = config['dp_config']['clientT']
    model_name = config['dpcr_model']['name']
    dataset_name = config['data_config']['dataset_name']

    aggrType = {"DPFedAvg": {},
                "SimpleMech": {"T": 1},
                "TwoLevel": {"kOrder": bestK(clientT, lambda x: x ** 2)},
                "BinMech": {"kOrder": bestK(clientT, lambda x: 2 ** x)},
                "Honaker": {"kOrder": bestK(clientT, lambda x: 2 ** x)},
                "FDA": {"kOrder": bestK(clientT, lambda x: (2 ** x - 1))},
                "BCRG": {"kOrder": bestK(clientT, lambda x: (2 ** x - 1))},
                "ABCRG": {"kOrder": bestK(clientT, lambda x: (2 ** x - 1))},
                "kTCR_k2": {"T": clientT},
                "kTCR_k3": {"T": clientT},
                "kTCR_k5": {"T": clientT},
                "kTCR_k8": {"T": clientT},
                "kTCR_k10": {"T": clientT}
                }
    dtArgs = {"MNIST": {'model__name': 'CNN_Mnist'},
              "FashionMNIST": {'model__name': 'CNN_Mnist'},
              "CIFAR10": {'model__name': 'WideResnet10_2'},
              }

    config['dpcr_model']['args'].update(aggrType[model_name])
    if config['model_config']['name'] is None:
        config['model_config']['name'] = dtArgs[dataset_name]['model__name']
    runFLDPCR(config)

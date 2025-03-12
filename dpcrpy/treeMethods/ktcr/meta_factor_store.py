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
# Used to store the meta-factors that have already been computed,
# thus achieving performance enhancement via pre-computation.
##########################################################################
import json
import os

import numpy as np


class MetaFactor:
    def __init__(self, alpha=None, beta=None, H=None):
        self.params = {}
        self.info = {}

        init_params = {'alpha': list(alpha), 'beta': list(beta)}
        init_info = {'H': H}

        self.paramNames = init_params.keys()
        self.infoNames = init_info.keys()

        for name in self.paramNames:
            if init_params.get(name) is not None:
                self.params[name] = init_params[name]

        for name in self.infoNames:
            if init_info.get(name) is not None:
                self.info[name] = init_info[name]

    def get(self, key):
        if key in self.paramNames:
            return self.params[key]
        if key in self.infoNames:
            return self.info[key]

    def set(self, key, value):
        if key in self.paramNames:
            self.params[key] = value
        if key in self.infoNames:
            self.info[key] = value

    def to_dict(self):
        return {
            'params': self.params,
            'info': self.info
        }

    def __repr__(self):
        return f"MetaFactor(params={self.params}, info={self.info})"


class MetaFactorStore:
    def __init__(self, file_path=None):
        if file_path is None:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.file_path = os.path.join(current_directory, 'data.json')
        else:
            self.file_path = file_path
        self.dataStore = {}
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                self.dataStore = json.load(file)
            self._convert_inf(encode=False)

    def save(self, output_file_path=None):
        if output_file_path is None:
            output_file_path = self.file_path
        self._convert_inf(encode=True)
        with open(output_file_path, 'w') as file:
            sjson = json.dumps(self.dataStore, indent=4)
            file.write(sjson)
        self._convert_inf(encode=False)

    def _convert_inf(self, encode):
        def convert_value(val):
            if isinstance(val, list):
                return [convert_value(x) for x in val]
            elif encode:
                return -1 if val == np.inf else val
            else:
                return float('inf') if val == -1 else val

        for k, v in self.dataStore.items():
            if 'gTheta' in v:
                self.dataStore[k]['gTheta'] = convert_value(v['gTheta'])

    def __contains__(self, keys):
        if not isinstance(keys, tuple) or (2 > len(keys) > 3):
            raise ValueError("必须提供 (k, h) 或 (k, h, N) 形式的键")
        if len(keys) == 3:
            k, h, N = (f'k_{keys[0]}', f'h_{keys[1]}', f'N_{keys[2]}')
            if k not in self.dataStore:
                return False
            if h not in self.dataStore[k]['meta']:
                return False
            if N not in self.dataStore[k]['meta'][h]:
                return False
        elif len(keys) == 2:
            k, h = (f'k_{keys[0]}', f'h_{keys[1]}')
            if k not in self.dataStore:
                return False
            if h not in self.dataStore[k]['meta']:
                return False
        return True

    def __getitem__(self, keys):
        if not isinstance(keys, tuple) or (2 > len(keys) > 3):
            raise ValueError("必须提供 (k, h) 或 (k, h, N) 形式的键")
        if len(keys) == 3:
            k, h, N = (f'k_{keys[0]}', f'h_{keys[1]}', f'N_{keys[2]}')
            try:
                data = self.dataStore[k]['meta'][h][N]
                return MetaFactor(**data['params'], **data['info'])
            except KeyError as e:
                raise ValueError(f"查询参数 {e} 无效")
        elif len(keys) == 2:
            k, h = (f'k_{keys[0]}', f'h_{keys[1]}')
            try:
                ls = list(self.dataStore[k]['meta'][h].values())
                return [MetaFactor(**data['params'], **data['info']) for data in ls]
            except KeyError as e:
                raise ValueError(f"查询参数 {e} 无效")

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple) or len(keys) != 3:
            raise ValueError("必须提供 (k, h, N) 形式的键")
        if not isinstance(value, MetaFactor):
            raise ValueError("值必须是 MetaFactor 类型")

        k, h, N = (f'k_{keys[0]}', f'h_{keys[1]}', f'N_{keys[2]}')
        if k not in self.dataStore:
            self.dataStore[k] = {'meta': {}}
        if 'meta' not in self.dataStore[k]:
            self.dataStore[k]['meta'] = {}
        if h not in self.dataStore[k]['meta']:
            self.dataStore[k]['meta'][h] = {}

        self.dataStore[k]['meta'][h][N] = value.to_dict()

    def set_theta(self, k, GTheta, gTheta):
        k = f'k_{k}'
        if k not in self.dataStore:
            self.dataStore[k] = {'meta': {}}
        self.dataStore[k]['GTheta'] = list(GTheta)
        self.dataStore[k]['gTheta'] = list(gTheta)

    def get_GTheta(self, k):
        k = f'k_{k}'
        return self.dataStore[k]['GTheta']

    def get_gTheta(self, k):
        k = f'k_{k}'
        return self.dataStore[k]['gTheta']

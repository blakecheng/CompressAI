# Copyright 2020 InterDigital Communications, Inc.
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
import os
import datetime
import logging


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_generic_signature(special_info):
    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    name = '{}_{}'.format(special_info, time_signature)
    root = os.path.join("../experiment",name)
    checkpoints_save = os.path.join(root,"checkpoints")
    figures_save = os.path.join(root, 'figures')
    tensorboard_runs = os.path.join(root, 'tensorboard')
    
    makedirs(checkpoints_save)
    makedirs(figures_save)
    makedirs(tensorboard_runs)

    return {"checkpoints_save": checkpoints_save,
        "figures_save":figures_save,
        "tensorboard_runs": tensorboard_runs
    }



# def logger_write(logpath):
#     logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
#                     filename=logpath)

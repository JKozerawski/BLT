# Copyright (c) Microsoft Corporation.
# Copyright (c) Jedrzej Kozerawski
# Licensed under the MIT license.

python3 main.py --config ./config/ImageNet_LT/stage_1.py --no_sampler --no_hallucinations --gpu 0
python3 main.py --config ./config/ImageNet_LT/stage_2.py --gpu 0

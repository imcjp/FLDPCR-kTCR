# FLDPCR-kTCR
### Code for "Enhancing Federated Learning with Differentially Private Continuous Data Release via k-Ary Trees"

This repository contains the code for the paper **"Enhancing Federated Learning with Differentially Private Continuous Data Release via k-Ary Trees"**. In this work, we propose a novel differential privacy continuous data release (DPCR) model, called the k-ary tree-based DPCR (kTCR) model.

Our approach leverages the open-sourced [**FLDPCR framework**](https://github.com/imcjp/FLDPCR), building upon it to introduce k-ary trees for constructing the release strategy. We further improve the model's performance by minimizing errors using Variance Optimal Estimation (VOE) and privacy budget allocation algorithms. The kTCR model significantly enhances learning accuracy compared to state-of-the-art DPCR models.

### Requirements

The source code requires **Python 3**. A list of required packages is detailed in [**requirements.txt**](requirements.txt) and [**setup.bat**](setup.bat), which includes:

- `torch==2.5.1+cu118`
- `torchvision==0.20.1+cu118`
- `torchaudio==2.5.1+cu118`
- `scipy`
- `tqdm`
- `opacus`

### Instructions

1. The proposed kTCR model is implemented in the [**ktcr**](dpcrpy/treeMethods/ktcr) module.
2. The kTCR model is designed based on the theory of k-ary numbers, which is implemented in [**kary_math.py**](dpcrpy/treeMethods/ktcr/utils/kary_math.py). For detailed explanations, refer to [**kary_math.md**](dpcrpy/treeMethods/ktcr/utils/kary_math.md).
3. The FL-DPCR framework is implemented in the `fldpcr` folder, which includes the server-side code (`server.py`) for collaborators and the client-side code for participants (`priClient.py` for private FL and `flClient.py` for non-private FL). Additional helper code can be found in the `utils` folder.
4. We provide a script, `main.m`, to help users quickly implement FL-DPCR using our code. This script includes several DPCR models, such as SimpleMech, TwoLevel, BinMech, FDA, BCRG, ABCRG, and the proposed kTCR model. For the kTCR model with k-ary trees, use `kTCR_k{k}` in the script, where the predefined values of `k` are {2, 3, 5, 8, 10}.
5. The `dpcrpy` and `opacus_dpcr` folders are used to implement DPCR and private DPCR learning. These are derived from [**Opacus-DPCR**](https://github.com/imcjp/Opacus-DPCR).

We hope this project facilitates further research in Federated Learning and Differential Privacy. For any issues or contributions, feel free to open an issue or submit a pull request.

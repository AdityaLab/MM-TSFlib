# MM-TSFlib
MM-TSFlib is an open-source library for multimodal time-series forecasting based on [Time-MMD](https://github.com/AdityaLab/Time-MMD/) dataset. We achieve multimodal time series forecasting tasks, including both short-term and long-term, by integrating time series models and language models. Our framework is illustrated in the figure.

<div align="center">
    <img src="https://github.com/AdityaLab/MM-TSFlib/blob/main/lib_overview_00.png" width="500">
</div>

:triangular_flag_on_post:**News** (2024.06)  Preprocessing functions and preprocessed data to speed up the training process will be released soon

 
## Usage

1. Install environment, execute the following command.

```
pip install -r environment.txt
```

2. Prepare Data. Our dataset is [Time-MMD](https://github.com/AdityaLab/Time-MMD/) dataset.
We provide preprocessed data in the ./data folder to accelerate training, particularly simplifying the text matching process.

## Citation

If you find this repo useful, please cite our paper.

```
@misc{liu2024timemmd,
      title={Time-MMD: A New Multi-Domain Multimodal Dataset for Time Series Analysis}, 
      author={Haoxin Liu and Shangqing Xu and Zhiyuan Zhao and Lingkai Kong and Harshavardhan Kamarthi and Aditya B. Sasanur and Megha Sharma and Jiaming Cui and Qingsong Wen and Chao Zhang and B. Aditya Prakash},
      year={2024},
      eprint={2406.08627},
      archivePrefix={arXiv},
      primaryClass={id='cs.LG' full_name='Machine Learning' is_active=True alt_name=None in_archive='cs' is_general=False description='Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:
hliu763@gatech.edu
## Acknowledgement

This library is constructed based on the following repos:

https://github.com/thuml/Time-Series-Library/

### Explaining  Image Classification through Knowledge-aware Neuron Interpretation

This repository contains the code for the paper "Explaining  Image Classification through Knowledge-aware Neuron Interpretation".



#### Experiments for Model Prediction Explanation

##### Data Preparation

The per data results (per_data_results.json) which  running each data individually can be cloned from https://drive.google.com/file/d/1jQavR3Rza6Hjw1EQ163imV15Tm5g8zv6/view?usp=sharing, and unzip it to the folder: Model_Prediction_Explanation/data

##### False Prediction

```
python Model_Prediction_Explanation/false_prediction.py
```

##### True Prediction

```python
python Model_Prediction_Explanation/true_prediction.py
```





#### Experiments for  Compositional Explanation

##### Using Concept Filtering

The complete results can be cloned from https://drive.google.com/file/d/15xbXfjeVT-eFMtIxN9rYI-KmG3a4qy3q/. 





#### Experiments for Model Manipulation

##### Data and Code Preparation

- The dataset (dataset.zip) can be cloned from https://drive.google.com/file/d/1rmGSvUnsyxXu6jfqB9w85XYvd8VTt_Lj/view?usp=sharing, and unzip it to the folder: Model_Manipulation/model_manipulation
- Replace "/site-packages/torchvision/models/resnet.py" in your environment with "Model_Manipulation/replace_code/resnet.py"

##### Disabling Positive Neurons

```python
python Model_Manipulation/model_manipulation/run_disable_pos_neurons.py
```

##### Disabling Negative Neurons

```python
python Model_Manipulation/model_manipulation/run_disable_nega_neurons.py
```






#### Dependencies

- `pyeda`
- `torch==1.4.0`
- `torchvision==0.4.2`
- `pandas==0.25.3`
- `tqdm==4.30.0`
- `imageio==2.6.1`
- `scipy==1.4.1`
- `matplotlib==3.1.3`
- `Pillow==7.0.0`
- `seaborn==0.9.0`
- `scikit-image==0.16.2`
- `pyparsing==2.4.6`
- `pyeda==0.28.0` 
- `pycocotools==2.0`

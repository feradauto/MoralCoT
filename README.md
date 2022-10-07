#  When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment

This is the repo for: "[When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment](https://arxiv.org/abs/2210.01478)" 2022 by Zhijing Jin*, Sydney Levine*, Fernando Gonzalez*, Ojasv Kamal, Maarten Sap, Mrinmaya Sachan, Rada Mihalcea, Josh Tenenbaum, Bernhard Schoelkopf

The dataset can be found [here] (https://github.com/feradauto/MoralCoT/blob/main/input_data/complete_file.csv)

**models** Contains the scripts to get the model predictions using GPT3 and baseline predictions  
**extra_analyses** Scripts with extra analyses. E.g. Domain features evaluation, price estimation, dogmatic score.  
**input_data** Contains the dataset ("complete_file.csv") and costs of items estimated by humans  


## Instructions to run the models

### Installation

1. `conda create -n moralcot python=3.7`
2. `conda activate moralcot`
3. `pip install -r requirements.txt`
4. `export base_folder=path_to_the_project`
5. `export OPENAI_API_KEY=your_gpt3_key`  necessary to query GPT3

### Generating predictions
To generate the predictions for all the models run:
```bash
`./main_models/paraphrases/run_models_ensemble.sh` to generate the predictions with all the paraphrases
```

MoralCoT uses OPENAI API
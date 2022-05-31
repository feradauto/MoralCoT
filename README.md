##  When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment

This is the repo for: "[When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment](https://drive.google.com/file/d/1hB_yStLu52sGUYOVwq1aQ8O-6IeqR12g/view)" 2022 by Zhijing Jin*, Sydney Levine*, Fernando Gonzalez*, Ojasv Kamal, Maarten Sap, Mrinmaya Sachan, Rada Mihalcea, Josh Tenenbaum, Bernhard Schoelkopf

The dataset is in /input_data/complete_file.csv

**/models** Contains the scripts to get the model predictions using GPT3 and baseline predictions
**/extra_analyses** Scripts with extra analyses. E.g. Domain features evaluation, price estimation, dogmatic score.
**/input_data** Contains the dataset ("complete_file.csv") and costs of items estimated by humans


### Instructions to run the models

1. `conda create -n moralcot python=3.7`
2. `conda activate moralcot`
3. `conda install pip`
4. `pip install -r requirements.txt`
5. `export base_folder=path_to_the_project`
6. `export OPENAI_API_KEY=your_key_gpt3`  necessary to query GPT3
7. `./main_models/run_models.sh` to generate the performance table

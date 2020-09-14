## Main code for the NLPCC submission.


### Requirements
+ Python >= 3.7.0
+ Run ```pip -r requirements.txt``` to install the other requirements.

### Data:
+ The original data is at ./data/dataset and it contains three json files: train.json, test.json, valid.json.
+ These jsons are extracted from our annotation platform directly.

### Preprocess
+ ```cd scripts``` to enter the code folder.
+  ```python preprocess.py``` to generate the data from original data file.

### Train & Evaluate
+ Execute ```sh run_all.sh``` to see the train and evaluation result of joint full model in output.log.
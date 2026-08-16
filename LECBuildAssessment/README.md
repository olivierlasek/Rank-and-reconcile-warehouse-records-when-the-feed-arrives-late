# Rank and reconcile warehouse records when the feed arrives late

An agent that monitors a live transaction stream and a warehouse data feed that arrives asynchronously, sometimes hours later. This agent decides which transactions to trust, flags inconsistencies it cannot resolve, and ranks them by confidence.

## How the code works

So for this project, I thought it would be a fun idea to train a neural network to identify the discrepencies. I went with the Random Forest Classifier using `scikit-learn` which is the model I think best represents what we're trying to do here. It will create a large number of decision trees and then decide which decision to make based on the `features` I have chosen to train it on. 

I generated a bunch of warehouse and transaction data using both real data and synthesised data (with the help of claude for ideas). 

* `train.csv` is for training the NN 
* `test.csv` is for testing the NN for its accuracy, precision, etc. 
* `val.csv` is used for testing individual lines of data for the `main.ipynb`. 
* `main.ipynb` contains all the imports, code, and where you can run a specific line of your choosing (+ a plot)

Then once everything is trained, you can test a specific line you want in `val.csv`. This will give you: the raw data, the final decision (+ probability/condfidence), and all the reasoning behind the NN's choice as proportions of which paramaters mattered to its final decision. 

## How well does it work? 

While choosing many different random seeds, I saw that the accuracy was always around 99% which is obviously really good, especially without any hyperparamater optimisations and etc. All the other classifications were really high and the NN always seems very confident with its answer as the probability is always close to 1. 

## Improvements for next time

* The obvious improvement is more training + testing data as per usual. 
* Fine-Tuning the hyperparameters such as `n_estimators` and using cross validation to fine-tune even more. 
* More preprocessing of the data to make sure everything is normalised, although, I don't think this is necessary. 


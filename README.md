# Bixente Grandjean
### Case Study

---
### Tree
```
.
└── case_study/
    ├── data/
    │   ├── dataset/
    │   │   └── should contain the match data
    │   ├── distributions/
    │   │   └── duration distriburion per action
    │   ├── durations/
    │   │   └── durations values per action
    │   ├── y_infs/
    │   │   └── inferior boundary of normalised gait per action
    │   └── y_sups/
    │       └── superior boundary of normalised gait per action
    ├── miscelaneous/
    │   └── exploration notebooks and api draft
    ├── models/
    │   └── latest model
    ├── src/
    │   ├── tools/
    │   │   └── models/
    │   │       └── the different models modules
    │   └── main -> the api app
    ├── Dockerfile
    ├── ReadMe.md
    └── requirements.txt

```

### Method

#### 1. Exploration

Data is composed of 2 files with similar yet different data (1 has less offensive actions than 2)
Walk actions have a longer duration compared to run ones.
Walk and run actions represent a vast majority of the actions.
Both files represent around 10 minutes games.
Acceleration norm is quite hard to manipulate due to the lack of direction.

#### 2. Approaches

First approach **LegModel**, try to recreate a basic leg model (assuming player's movement in 1D, all actions have similar gaits, only forward).
Quite difficult to put in place but useful to understand the behavior of the sensor over a gait.

Second approach, using RNN. We train a model to predict the next action given the last n actions.
Due to the lack of memory, the RNN could predict several shot in a row.

Third approach, split the problem in 3 parts :
- predict the next action **ActionModel** using LSTM
- predict the duration of that action (using a smoothened probability density) of the provided data.
- predict the acceleration norms array (using a scaled average gait for each action)

#### 3. - 4. Re-creation

Notebooks in **miscelanous/** contains the models drafts and short qualitative tests.
**DataExploration** notebook allows to explore (obviously) the dataset but also to generate data for Duration and Norm models.

Training was done using **ActionModel** notebook with the exact parameters than are in the notebook.
With more time, it could be interesting to try to tweak a few parameters.
No validation data was used because of the lack of data. Hence, only the loss in calculated for the training.

Loss is calculated this way :
- for each susequence (let *walk-walk-run*), we extract the real distribution for the next action.
- the model tries to match as much as possible that distribution (normalized for a total probability of 1)


Furthermore, the ActionModel predicts as follows :
- it gets a sequence of actions as input
- it outputs a 1D tensor representing the probability of predicting any of the actions
- the model choses at random (using the tensor for the weigths) an action

Then, the DurationModel predicts :
- it gets an action as input
- it choses at random the duration (using the real data distribution durations of the given action)

Finally, it scales the "average gait" for that given action to the received duration (adding some variation within the standard deviation range)

This process continues as long as needed to reach the requested amount of time.

#### 5. API

*Note : the API isn't available anymore*

An API is available at :
```
http://sandbix.fr:1234/predict
```
Usage example :
```
curl -X POST -H 'Content-Type: application/json' -d '{"duration": 10}' 'http://sandbix.fr:1234/predict'
```
With duration : the duration (in minutes) of the predicted game

To run it locally :
```
docker build . -t case_study
docker run -d -p 1234:8000 case_study
```

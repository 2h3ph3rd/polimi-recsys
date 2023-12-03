# POLIMI - Recommender Systems Challenge

My solution to the Recommender Systems challenge at Polimi for the academic year 2022/2023. 

It's important to mention that a significant portion of the code and algorithms has been taken from the official course's repository.

https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi

## Data

`data.zip` contains the offical files given for the competition.
Extract it inside the root folder of the project.

### interactions_and_impressions.csv

Contains the training set, describing implicit preferences expressed by the users.

- user_id : identifier of the user
- item_id : identifier of the item (TV series)
- impression_list : string containing the items that were present on the screen when the
- data : "0" if the user watched the item, "1" if the user opened the item details page.

### data_ICM_length.csv

Contains the number of episodes of the items. TV series may have multiple episodes.

- item_id : identifier of the item
- feature_id : identifier of the feature, only one value (0) exists since this ICM only contains the feature "length"
- data : number of episodes. Some values may be 0 due to incomplete data.

### data_ICM_type.csv

Contains the type of the items. An item can only have one type.

All types are anonymized and described only by a numerical identifier.

- item_id : identifier of the item
- feature_id : identifier of the type
- data : "1" if the item is described by the type

### data_target_users_test.csv

Contains the ids of the users that should appear in your submission file.

The submission file should contain all and only these users.

## What we know

- The type of some items (tv/film)
- Some of the user interactions
- How many elements is composed of each item

## History

| score   | recommenders                                                                            |
| ------- | --------------------------------------------------------------------------------------- |
| 0.0508  | RP3betaCBF(alpha=0.7, beta=0.3)                                                         |
| 0.04535 | RP3betaCBF(alpha=0.7, beta=0.3) + MultiThreadSLIM_ElasticNet(alpha=0.05, l1_ratio=0.08) |
| 0.04930 | RP3betaCBF(alpha=0.7, beta=0.3) + MultiThreadSLIM_ElasticNet(alpha=1, l1_ratio=0.1)     |
| 0.04934 | RP3betaCBF(alpha=0.85, beta=0.35)                                                       |

### Hybrid

| score   | recommenders |
| ------- | ------------ |
| 0.04425 | P3Alpha      |

## To Do

- cutoff [5, 10, 15]

## Experiments

### 26/12

ItemKNN + Hybrid(ItemKNN, P3Alpha)

### 27/12

PureSVD + Hybrid(ItemKNN, P3Alpha, PureSVD)

### 28/12

P3Alpha vs RP3Beta
CBC version (?)

# RecSys Challenge 2022

Solution for recommender systems challenge at Polimi for the accademic year 2022/2023.

## Files

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
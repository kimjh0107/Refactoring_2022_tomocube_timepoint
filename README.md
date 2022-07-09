# [REFACTORING] Tomocube Timepoint Classification 
- Author: JongHyun Kim
- Date: 2022-06-16

## Objective 
Cell classification by each timepoint using every data 

### Data 

|date|Patient ID|timepoint
|:---:|:---:|:---:|
|20220405|1-1|timepoint_1|
|20220421|2-1|timepoint_1|
|20220422|3-1|timepoint_1|
|20220422_2|2-2|timepoint_2|
|20220425|3-2|timepoint_2|
|20220502|4-1|timepoint_1|
|20220510|4-2|timepoint_2|
|20220519|5-1|timepoint_1|
|20220523|5-2|timepoint_2|
|20220520|6-1|timepoint_1|
|20220523_2|6-2|timepoint_2|
|20220526|7-1|timepoint_1|
|20220527|7-2|timepoint_2|
|20220526_2|8-1|timepoint_1|
|20220530|8-2|timepoint_2|
|20220602_3|9-1|timepoint_1|
|20220603_2|9-2|timepoint_2|
|20220603_3|10-1|timepoint_1|
|20220607_3|10-2|timepoint_2|
|20220607_2|11-1|timepoint_1|
|20220608|11-2|timepoint_2|

# Leave one out dataset 
1) blind test_1 ID: patient 2 
2) blind test_2 ID: patient 3 
3) blind test_3 ID: patient 4 
4) blind test_4 ID: patient 5 
5) blind test_5 ID: patient 6 
6) blind test_6 ID: patient 7
7) blind test_7 ID: patient 8 
8) blind test_8 ID: patient 9
9) blind test_9 ID: patient 10 
10) blind test_10 ID: patient 11

- timepoint 1: 20220405, 20220421, 20220422, 20220519, 20220520, 20220526 ,20220526_2, 20220602_3, 20220603_3, 20220607_2
    - patient ID: 1-1, 2-1, 3-1, 5-1, 6-1, 7-1, 8-1, 9-1, 10-1, 11-1
    - CD4: 1106, CD8: 1134
    
- timepoint 2: 20220422_2, 20220425, 20220523, 20220523_2, 20220527, 20220530, 20220603_2, 20220607_3, 20220608
    - patient ID: 2-2, 3-2, 5-2, 6-2, 7-2, 8-2, 9-2, 10-2, 11-2 
    - CD4: 1082, CD8: 1020


- patient4(test set): 20220502, 20220510 
    - timepoint 1: CD4: 34, CD8: 24
    - timepoint 2: CD4: 71, CD8: 59


### Blind test Result 

### 1. Standardization result 
|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blin_test_2|patient 3|CD8|92.20|90.90|92.06|0.300
|""|""|CD8_blindtest|60.76|54.31|61.76|0.80
|""|""|CD4|60.36|60.12|60.18|0.78
|""|""|CD4_blindtest|41.43|37.18|39.58|1.366
||
|blin_test_3|patient 4|CD8|95.48|93.30|93.98|
|""|""|CD8_blindtest|27.40|26.57|48.19|
|""|""|CD4|75.95|60.80|66.82|
|""|""|CD4_blindtest|34.63|29.07|37.14|
||
|blin_test_4|patient 5|CD8|93.71|93.46|93.51|0.275
|""|""|CD8_blindtest|73.92|72.61|73.48|0.50
|""|""|CD4|67.73|58.07|66.67|0.557
|""|""|CD4_blindtest|83.78|92.65|87.19|0.31

### 2. test_df = (test_df) / (10000)
|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blin_test_2|patient 3|CD8|93.55|92.20|93.46|0.27
|blin_test_2|patient 3|CD8_blindtest|49.19|46.77|51.96|1.027
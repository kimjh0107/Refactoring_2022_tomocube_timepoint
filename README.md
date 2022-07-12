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
1) blind test_1 ID: patient 2 (n=19)
2) blind test_2 ID: patient 3 (n=102)
3) blind test_3 ID: patient 4 (n=83)
4) blind test_4 ID: patient 5 (n=396)
5) blind test_5 ID: patient 6 (n=473)
6) blind test_6 ID: patient 7 (n=253)
7) blind test_7 ID: patient 8 (n=378)
8) blind test_8 ID: patient 9 (n=68)
9) blind test_9 ID: patient 10 (n=201)
10) blind test_10 ID: patient 11 (n=203)

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

### 3. individual normalization - with dropout 0.5 
|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blin_test_2|patient 3|CD8|84.89|82.90|84.58|0.48
|""|""|CD8_blindtest|48.50|46.46|50.98|0.77
|blin_test_3|patient 4|CD8||||
|""|""|CD8_blindtest|56.78|38.80|72.29|0.63
|blin_test_4|patient 5|CD8||||
|""|""|CD8_blindtest|72.99|72.85|72.47|0.81
|blin_test_5|patient 6|CD8||||
|""|""|CD8_blindtest|97.12|89.87|90.70|0.303


### 4. individual normalization - no dropout 
|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blind_test_1|patient 2|CD8|91.56|89.27|91.44|0.35
|""|""|CD8_blindtest|52.22|47.37|47.37|0.85
|blind_test_2|patient 3|CD8|83.16|81.89|82.71|0.49
|""|""|CD8_blindtest|50.58|47.37|52.94|0.79
|blind_test_3|patient 4|CD8|91.83|86.63|89.35|0.46
|""|""|CD8_blindtest|88.14|51.12|75.90|0.54
|blind_test_4|patient 5|CD8|92.67|92.46|92.43|0.29
|""|""|CD8_blindtest|72.49|72.37|71.97|0.91
|blind_test_5|patient 6|CD8|95.88|92.31|92.13|0.32
|""|""|CD8_blindtest|96.96|90.19|91.12|0.27
|blind_test_6|patient 7|CD8|93.72|89.87|90.45|0.40
|""|""|CD8_blindtest|95.91|77.88|81.03|0.44
|blind_test_7|patient 8|CD8|93.56|92.39|93.05|0.35
|""|""|CD8_blindtest|92.90|73.38|71.69|0.4933
|blind_test_8|patient 9|CD8|93.77|90.42|92.20|0.31
|""|""|CD8_blindtest|86.15|49.74|79.41|0.44
|blind_test_9|patient 10|CD8|94.42|94.05|94.63|0.25
|""|""|CD8_blindtest|83.24|74.57|81.59|0.60
|blind_test_10|patient 11|CD8|78.77|72.82|79.90|0.80
|""|""|CD8_blindtest|93.26|89.71|92.61|0.27


|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blind_test_1|patient 2|CD4||||
|""|""|CD4_blindtest||||
|blind_test_2|patient 3|CD4||||
|""|""|CD4_blindtest||||
|blind_test_3|patient 4|CD4||||
|""|""|CD4_blindtest||||
|blind_test_4|patient 5|CD4||||
|""|""|CD4_blindtest||||
|blind_test_5|patient 6|CD4||||
|""|""|CD4_blindtest||||
|blind_test_6|patient 7|CD4||||
|""|""|CD4_blindtest||||
|blind_test_7|patient 8|CD4||||
|""|""|CD4_blindtest||||
|blind_test_8|patient 9|CD4|55.37|57.29|58.41|0.67
|""|""|CD4_blindtest|68.44|50.48|70.30|0.59
|blind_test_9|patient 10|CD4|55.69|57.21|64.56|0.65
|""|""|CD4_blindtest|70.31|84.24|46.56|0.71
|blind_test_10|patient 11|CD4|66.96|61.89|61.50|0.64
|""|""|CD4_blindtest|53.18|42.35|70.90|0.611





### 5. Standardization result - with dropout 0.5 학습 x 
|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blin_test_2|patient 3|CD8|50.00|51.87|51.87|2.01
|""|""|CD8_blindtest|50.00|47.06|47.06|2.22

### 6. Standardization 모든 데이터의 값 result - with dropout x  학습 x 
|blind_test|exclusion_Patient_ID|test_dataset|AUROC|AUPR|ACC|loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|blin_test_2|patient 3|CD8|50.00|51.87|51.87|2.01
|""|""|CD8_blindtest|50.00|47.06|47.06|1.84
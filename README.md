# Stroke-Patient-Prediciton

We all know that stroke is a disease that easily leads us to death. Stroke can attack anyone regardless their gender, age, or any kind of attribute. It is basically caused by blood blockage in some brain parts, thus it precludes the blood to supply our brain, or in other words our brain will stop obtaining oxygen and nutrients. In minutes, death will come to us with no mercy.

In this article, I will demonstrate how to predict a person whether he or she will experience stroke or not using Machine Learning. How beneficial is this if a machine can diagnose you, or your beloved ones earlier. For those who are still a beginner in this field, please do not discourage to read this article.

# STEP 1: Grab your dataset

You can download dataset in Kaggle.com 

# STEP 2: Import libraries

![image](https://user-images.githubusercontent.com/84617244/145663210-aca3d139-ed94-4366-bef1-4537be905a96.png)

# STEP 3: Clean your dataset

Cleaning data means to clean the data from typos or removing outliers if exist and any job to make the data is more understandable by the machine to model. Of course machine only understands data if it is in numerical form. We can see it further how to transform the categorical value into numerical value. First we have to take a look to see what happens in our data

![image](https://user-images.githubusercontent.com/84617244/145662295-4c77f04b-eca9-45e3-ac80-cd4bac82d622.png)

In that picture you can see there is a lot of NaN values which indicate that our datasets has to be cleaned further. We can clearly see as well our data comprises both categorical and numerical value which require different treatment.
We have to spot any mistakes in our data, Gender has only 2 kinds : 'Male' and 'Female', you can drop the data if it is neither.

![image](https://user-images.githubusercontent.com/84617244/145671977-d314acae-ff8e-4240-a9ce-fc6c7f8a9e29.png)

The interesting part is that our feature 'Smoking Status' has 'Unknown' value for more than 1000 data, it is better to drop the feature than the data since 20% data lie in 'Unknown' category and it will not make sense to feed lots of vague data to the Machine Learning.
Afterwards, we have to take care of any missing data,

![image](https://user-images.githubusercontent.com/84617244/145672788-8d86f1b0-2248-477a-ba5d-19553395cd5c.png)

BMI feature has 201 missing data, that we will treat them using KNN imputer.

![image](https://user-images.githubusercontent.com/84617244/145674468-0bd0bb48-9c72-47f5-b6e7-a935f3fe1cc1.png)

I don't recommend you to use mean, mode, and median imputer since it only depends on one variable without considering others, while KNN Imputers will work more effective in this case, and also because this is a healthcare dataset, one has to carefully treat the data even to fill the missing value.

The next step is to transform our categorical value into numerical value. In this case, I perform one-hot encoding technique. Why don't use ordinal transformation?
In my opinion, if we perform such transformation, we will introduce 'order' in our categorical feature, so it is unsuitable in our case. Don't forget to drop the first variable during the transformation, because we don't want to make our features explain other features (multicollinearity), our independent variables have to have no correlation between one another otherwise this will cause our ML algorithms perform poorly.

![image](https://user-images.githubusercontent.com/84617244/145675592-3ee00cdd-a20f-48eb-b139-a2417775b951.png)

# STEP 4 : Exploratory Data Analysis

It is a must-step to work with especially if you want to work in data analysis domain.
However I don't want to go over it too detailed, but it can contain univariate or bivariate analysis. You can use histogram, piechart, or pair plot to demonstrate what happens with your data. Here is the histogram of the BMI and average glucose level

![image](https://user-images.githubusercontent.com/84617244/145675829-66ef8ca0-acba-4398-aa81-1f016cb56330.png)

At a glance you can now know that our dataset is largely composed by underweight and overweight, but mostly is overweight. Even there is a person whose BMI is above 80.
You can remove that data (if you really know that the patients misstype the BMI) but in this case I do not perform this technique since outliers are sometimes also needed for further observation in healthcare field.
The next is average glucose level

![image](https://user-images.githubusercontent.com/84617244/145676316-96111463-e467-4a6b-96b9-e22c3eaa0c71.png)

There seems to have two peaks, for normal avg glucose level falls between 140-180 mg/dL and those whose above that will be indicated as prediabetes or diabetes.
It is enough with univariate analysis, let's move on to multivariate analysis. Multivariate analysis is done to detect the multicollinearity, commonly it uses Pearson's correlation coefficient if the data is numerical. If the value nears 1 or -1 then it is strongly correlated, but if it nears 0 the correlation is weak. Apparently ours has no correlation on one each other, then it is good to go to the next step.

![image](https://user-images.githubusercontent.com/84617244/145676465-b658c433-d905-4427-817f-258e01f5c106.png)

# STEP 5 : Feature Selection

As the name suggests, feature selection is a method to select feature which is important to the creation of the model. As we discussed earlier, you can remove the independent variables which are strongly correlated with each other, and variables which are very unnecessary ('ID', or 'name' is unnecessary to the model)

Why do we need this step? Because we want to make our model simpler for not being interfered by lots of variables to get better accuracy.
You can drop the ID first, and as we know our numerical variables has no multicollinearity then it is good to save all numerical variables. However, we have to make our hands dirty to analyze the categorical variables. We employ chi-squares to analyse the importance of each categorical feature towards the target variable which is stroke.

![image](https://user-images.githubusercontent.com/84617244/145676825-fb8539e2-8310-4d2b-905a-f3162f3a996d.png)

What is shown is the p-value not the chi-squares value. The higher the p-value the more we believe that these features are not important. Thus we can drop all of these value which are higher than 0.05

# STEP 6 : Splitting Train Test Data

Splitting train test data is necessary for any ML tasks. I use 90% of my data to be used as training data and the rest is for my validation data. But please keep in mind that we have to observe whether our dataset is fairly balance or not. Thus to minimise the risk of imbalance data, we can use stratify train test split.

![image](https://user-images.githubusercontent.com/84617244/145678308-8950427a-fa6c-4782-826e-419f5703a4f7.png)

Stratify is used when our dataset is imbalance, most importantly in classification task. If I put train_size = 0.9 thus stratify will take 0.9 data of the majority and 0.1 of the minority, and the rest of all into the validation dataset. 

So you don't have to worry if your dataset will only include the majority class and the testing dataset will only  include the minority class

# STEP 7 : Treating Imbalance Data

Imbalance data can also contribute alot to the ML performance. If we introduce our ML with the majority data, thus the trained model will recognize more or overfit the majority data. This will impair our model performance. Therefore, balancing data is crucial in the classification tasks.

![image](https://user-images.githubusercontent.com/84617244/145678973-3b815666-746d-4cec-9b29-f5f1f1eab5dd.png)

Before balancing the data, we can clearly see that our dataset is highly imbalance. The 'non-stroke' group of people are having higher amount than the stroke group of people.
There are ways to get this done, you can either do oversample or undersample. I perform SMOTE algorithm to oversample the data. This is done after splitting train test dataset and be done in training dataset only

WHY? Because we don't want our testing dataset is leaked from anything, free from any interference. Thus we make our testing data as free as possible from any interference.

![image](https://user-images.githubusercontent.com/84617244/145678709-3e8a63ed-16d6-4fbd-b1c1-b73115c2d1cc.png)

# STEP 8 : Feeding the training data into the ML algorithm






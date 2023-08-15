Shouts out to Tuo, Vi and Tien for the moral and technical critiques given in creating this model! 

## Approach Key Elements with Underlying Assumptions
Our approach consists of 5 stages: acquiring domain knowledge, exploratory data analysis, feature engineering, modelling, and evaluation. Our motivations were driven by the potential utility provided from a model that classifies coordinates to help agriculturalists categorise land assets to make better-informed decisions in planning, farming, and logistics. 

Acquiring domain knowledge refers to understanding how different satellites captured data attributed to rice-farming activity. Sentinel 1 was found to capture radar backscatter of the terrain over time without interference from redundant physical elements such as clouds. Hence, it is selected as the data source. Moreover, capturing multiple snapshots of each coordinate is critical because rice farming activity generates temporal fluctuations that cannot be analysed from a single snapshot. Furthermore, we hypothesised that our solution would involve computer vision applied to sequences of images. 

Exploratory data analysis examined Sentinel 1 features sliced by labels. Figure 1 aggregates VV and VH over time, stratified by labels, which enables us to identify differentiable temporal patterns between rice and non-rice fields. Moreover, Figure 2 shows the overall distributions of features, and indicates that labels are differentiable by feature mean, variance, and range. Exploratory data analysis therefore infers that VV and VH features would be sufficient to differentiate rice and non-rice fields. 


<img width="364" alt="image" src="https://github.com/GaruChenk/Sentinel_1_ConvLSTM_Classifier/assets/73525913/96da8c46-515d-4d09-ae09-759531204af3">

<img width="357" alt="image" src="https://github.com/GaruChenk/Sentinel_1_ConvLSTM_Classifier/assets/73525913/382b6f64-8f90-47b3-9878-91b633af324a">

Our feature engineering avoided the aggregation of raw data. Instead, we stacked VV, VH, VV/VH tensors into RGB time lapses for each sampled coordinate. The result is a single predictor that expresses temporal and visual information of each coordinate.

Modelling focuses on implementing a deep learning model, which could analyse a sequence of images over time. According to our research, CNNs are suitable for extracting visual information, whilst LSTMs could extract temporal information from image sequences. As a result, our modelling combines both types of neural networks into a multi-layer ConvLSTM architecture. 

Evaluation refers to assessing model performance when generalised. Particularly, it involves performing cross-fold validation, evaluation metrics, and visualisations. The outputs from this stage assisted us in hyperparameter tuning. 

According to the approach introduced above, our assumptions are: 
•	The predictive task applies to a particular season; coordinates are only labelled as rice if they grew rice during the season of interest.
•	Some non-rice coordinates may be planting crops other than rice.
•	Field sizes can be constant across the different coordinates sampled, if the area of the smallest field is greater than the bounding box parameter. 

## Innovative and Unique Aspects of the Approach
We successfully implemented a deep learning model as a simplistic, yet effective solution to the problem. Our approach is generalised to perform any binary classification task involving Sentinel 1 data, given that labels for coordinates are available for supervised learning.

Furthermore, our feature engineering did not generate any domain-specific features, so our model could learn different visual-temporal patterns associated with a coordinate, because the predictor dataset contains rich information. Therefore, our approach could also be applied to classify other visual labels such as crop health or land type across different scenarios.

By only using 4 months of Sentinel 1 VV and VH data, we reduce model dependencies to a single instrument. It reduces the probability of collecting dirty inputs when using multiple data-collection instruments or sampling larger quantities of data. Our pre-processing can also be computed quickly and remain robust for scaling and transferability to other applications.  

## Target and Predictor Datasets
The target dataset is the labels provided in the ‘Class of Land’ column in the ‘Crop_Location_data.csv’ file provided. 

The raw predictor datasets are VV and VH bands derived from Sentinel-1-RTC, at a resolution of 10 metres per pixel, a box degree size of 0.008, and a time frame between 2022-04-01/2022-08-31. 

## Data Preparation
First, we downloaded raw Sentinel 1 VV and VH data from the Planetary Computer API and pickled them as a sampled list of tensors. This was done in a single batch capturing both train and test samples, where outlier and null value treatment was not necessary. 

After extracting VV and VH tensors from the Planetary Computer Sentinel-1-RTC API, we wrangled the dataset so that samples had identical data shapes. This was done through the reshapeData3D function in our notebook, which takes the dataset, and trims the length of each dimension to a specified pixel and instance length. For our model, we ensured that each sample had a 90x90 pixels, extrapolated over 24 time periods. 

The next step was computing the RGB tensor feature. This was done by stacking VV, VH and VV/VH tensors to form a time lapse of RGB images for each sample to support our modelling capabilities.

After that, we encoded our target labels on the training data using a standard label encoder to convert Rice and Non-Rice labels to 0 and 1. The dataset was then split into train, test and validation sets after feature engineering.

## Model Validation
Model validation involves analysing evaluation metrics and cross-fold validation performance. 

We held out 20% of the training data for validation purposes and observed loss and accuracy for both sets on each iteration to identify overfitting and underfitting issues. Figure 2 shows that variance between accuracy and validation accuracy reduced during training, leading to a slightly under-fitted model. After the initial fitting, we generated classification reports and confusion matrices based on model prediction on seen and unseen data to confirm that the model has been generalised. 

<img width="337" alt="image" src="https://github.com/GaruChenk/Sentinel_1_ConvLSTM_Classifier/assets/73525913/07cec0b3-f6fc-4e0a-91e3-e17b03a4d270">

To assess how well our model generalises, we performed a 5-fold CV. The cross-folds exhibited a 20% variance with a mean accuracy of 78%. These results suggest that small samples skew the model from generalising due to outlier patterns in some samples. Figure 2 supports this finding, which shows that rice VV, VH distribution exist as a subset within the distribution of non-rice fields.

This finding may be of concern when generalising our model to other locations, where a lack of sufficient or relevant training data may create difficulties for the model to discern labels. Therefore, the recommendation is to train the model on a large dataset with diverse geographies so that the model may learn universal features of rice fields that are transferrable globally. 


## Highest-Performing Features
According to feature engineering, RGB time series tensor feature exhibited the highest performance compared to aggregated and raw VV, VH, VGI.

## Model Selection
Our approach began with models that depended on aggregated VV, VH features. We engineered features such as RVI and achieved reasonable accuracy using support vector machines and tree-based classifiers. We then incorporated temporal features by stratifying annual VV, VH and RVI means by month, which expanded our feature set by a multiple of 12. Finally, we determined the optimal timeframe with the highest predictive accuracy through principal component analysis and recursive feature elimination. We reached 97% accuracy using support vector machines, but our models could not be trained on unaggregated raw data.

As a result, we pivoted towards a deep learning approach, where we started implementing CNNs on RGB features. After that, we added LSTM layers to our architecture and eventually converged on Conv-LSTM. Throughout this process, our feature engineering pivoted from aggregation to preservation of the raw dataset structure. Our final model, whilst less efficient, surpassed our original approach in terms of prediction accuracy and transferability to different applications. 

## Prediction Accuracy Score on the Unseen Data
Once the model was fitted, our prediction accuracy was 100% with a cross-permutation-entropy loss of 0.

## Most Critical Breakthrough for Score Improvement
The most critical breakthrough made was the transition in our approach from applying advanced analytics techniques to deep learning methodologies. This paradigm shift transformed the way we approached feature engineering and modelling. For example, instead of trying to extrapolate the dataset into a matrix, we relied upon RGB tensors that preserved variance between and across visual-temporal features.

## Challenges
The most challenging part was setting the right Planetary Computer API parameters to obtain suitable datasets. Scraping data from the API was time-consuming, and we had to use trial and error to determine the dataset's optimal bounding box and resolution.

## Inference Runtime and Optimising Efficiency
Our model processing time could be further optimised by adjusting the learning rate, the number of layers and batch size. 

Furthermore, we could also tune the pixel resolution of the dataset. Currently, each image is 90x90, which is expensive to process. We could either decrease pixel resolution and increase box size area or decrease pixel resolution to allow our model to process inputs faster, at the expense of information loss.
















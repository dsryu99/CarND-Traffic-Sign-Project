# **Traffic Sign Recognition**
---
### Build a Traffic Sign Recognition Project
### The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./my_examples/visualization.png "Visualization"
[image2]: ./my_examples/2000_1.png "Training Data Instance with Index 2000 and Label 1"
[image3]: ./new-images/1.jpg "Traffic Sign 1"
[image4]: ./new-images/2.jpg "Traffic Sign 2"
[image5]: ./new-images/3.jpg "Traffic Sign 3"
[image6]: ./new-images/4.jpg "Traffic Sign 4"
[image7]: ./new-images/5.jpg "Traffic Sign 5"
[image8]: ./my_examples/1_predicted.jpg "Predicted Traffic Sign 1"
[image9]: ./my_examples/4_predicted.jpg "Predicted Traffic Sign 4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one. The submission includes the project code. Here is a link to my [project code](https://github.com/dsryu99/CarND-Traffic-Sign-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3))
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the classes are distributed in the training, validation and test set.

![alt text][image1]

The following image is an example of the training data containing the German Traffic Signs.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a preprocessing step, I normalized the image data by using a min-max method because the gradient descent algorithm should converge to the optimum. If training data are mixed up with large-value and small-value data, it is very likely to diverge during the gradient descent processing.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is a basically LeNet architecture model and I added a dropout to avoid the overfitting of the model. My final model consisted of the following layers:

| Layer           		|     Description	        					|
|:----------------------:|:-------------------------------------------------:|
| Input            		| 32x32x3 RGB image   							|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU				       	|												|
| Max pooling	       	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 3x3     | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU                | |
| Max pooling         | 2x2 stride, valid padding, outputs 5x5x16   |
| Flatten             | outputs 400 |
| Fully connected    	| outputs 120|
| RELU                | |
| Fully connected     | outputs 84 |
| RELU                | |
| Dropout             | keep prob 0.5 |
| Fully connected     | outputs 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer. The learning rate is 0.001. The batch size, number of epochs and the keeping probability for the dropout is 50, 128 and 0.5 respectively.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.934
* test set accuracy of 0.919

The first architecture chosen was the LeNet architecture. It is a well-known model that can classify traffic signs well. Initially, the validation accuracy was around 0.7. Thus, I added the min-max normalization so that the gradient descent algorithm can converge to the optimum. Otherwise, the diversion can occur.
Next, in order to relieve the overfitting problem, I added the dropout using the keeping probability 0.5. I also increased the number of epochs from 10 to 50. By using those adjustments, I can acheive the validation set accuracy to be 0.93.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because the characters "stop" are all white colors. If they are black colors, the classification would be easier. The predicted class by the model is "Priority road" exemplified by this image ![alt text][image8]

The second, third and fifth images are correctly classified by the model.
The fourth image might be difficult to predict because the black color inside the triangle is not clear. The predicted class by the model is "General caution" exemplified by the image ![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:-----------------------------------------:|
| Stop Sign      		| Priority road   									|
| Road work     			| Road work 						  				|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Slippery road	      		| General caution			 				|
| Turn right ahead			| Turn right ahead      				|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is a very poor performance compared to that on the test set reaching 91%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is sure that this is a Priority road (probability of 1.0), but the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Priority road   									|
| 0.00     				| Speed limit (100km/h) 										|
| 0.00					| Speed limit (70km/h)											|
| 0.00	      			| Roundabout mandatory					 				|
| 0.00				    | Speed limit (80km/h)      							|

For the second image, the model is sure that this is a Road work (probability of 1.0), and the image does contain a Road work. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Road work   									|
| 0.00     				| Dangerous curve to the right 										|
| 0.00					| General caution											|
| 0.00	      			| Double curve					 				|
| 0.00				    | Bumpy road      							|

For the third image, the model is sure that this is a Right-of-way at the next intersection (probability of 1.0), and the image does contain a Right-of-way at the next intersection. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Right-of-way at the next intersection    		|
| 0.00     				| Beware of ice/snow  										|
| 0.00					| Speed limit (20km/h)											|
| 0.00	      			| Speed limit (30km/h)					 				|
| 0.00				    | Speed limit (50km/h)      							|


For the fourth image, the model is sure that this is a General caution (probability of 0.88), but the image does contain a slippery road. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.88         			| General caution   									|
| 0.12     				| Right-of-way at the next intersection 			|
| 0.00					| Pedestrians 											|
| 0.00	      			| Dangerous curve to the right					 				|
| 0.00				    | Road work       							|

For the fifth image, the model is sure that this is a Turn righ ahead (probability of 0.98), and the image does contain a Trun right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.98         			| Turn right ahead    									|
| 0.02     				| Ahead only  										|
| 0.00					| Go straight or left 											|
| 0.00	      			| Keep left 					 				|
| 0.00				    | Roundabout mandatory       							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

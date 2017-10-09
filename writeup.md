#**Traffic Sign Recognition** 

##Writeup 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/download.png "Visualization"
[image2]: ./image/download(1).png "image sample"
[image3]: ./image/download(2).png "5 images"
[image4]: ./image/download(3).png "predict result 1"
[image5]: ./image/download(4).png "predict result 2"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

As the dataset is in numpy.array format, I got the shape of the dataset by **.shape** property, also **max(label) - min(label) + 1** should output the number of classes, now we get the answer!
signs data set:

* The size of training set is **(34799, 32, 32, 3)**
* The size of the validation set is **(4410, 32, 32, 3)**
* The size of test set is **(12630, 32, 32, 3)**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

Wow! It seems we've got a lot of data here!

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many data are there for each label, and you can see some labels has very rich dataset but some are very poor, so the model may have a better peoformance on those with more labels.

![alt text][image1]

Here I also visualize an image from the training dataset and the label. Everything looks good!

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I can't decided whether I should convert the images to grayscale, since there are more info in colour image but grayscale can make model more concentrate, so I implement them both, and use a **is_gray** buffer to indicate that.
After the gray scale, I normalized the image data to make the model learn faster, or easier for weight init.
After the comparsion between gray and colour,I find colour have a better result,so I choose colour here.

Here is an example of a traffic sign image before and after grayscaling.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model mostly based on LeNet arch, but I've changed the network a little bit. The last layer of original LeNet arch has 84 neuron for 10 classes classification, but now we have 43 classes to classify, so I add more neuron on every fc layer since it may need more neuron for process the feature information.
I also add dropout for the model,but I did not add it to conv layer, because the number of feature map is too small, and dropout can make the feature extraction very unstable. 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6   	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16   	|
| flatten       		| outputs 400    								|
| Fully connected		| outputs 256									|
| RELU					|												|
| dropout				|												|
| Fully connected		| outputs 128									|
| RELU					|												|
| dropout				|												|
| Fully connected		| outputs 43									|
| Softmax				| etc.        									|




####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adaptive optimizer, Adam optimizer, batch size is 128 for a better performance and faster training. Epoch set to 28, because during my experiment I found that after 23 Epoch, the validation accurarcy changed slightly.learning rate set to 0.001,keep_prob set to 0.6,these hyperparamters adjusted according to the training performance.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9
* validation set accuracy of 96.6 
* test set accuracy of 94.3

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
My answer: originally, I used Lenet, but it only reach about 90% accurarcy.
* What were some problems with the initial architecture?
My answer: I've notice that, the final layer of LeNet has 84 unit for 10 classes, but for our problem, we have 43 classes here, so the problem may be the lack of hidden unit, so I add the number of unit for each hidden layer(include conv layer).
Then the model got 93%-94% accurarcy.but I think I can do better!
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
My answer: 
1.Then I realize that maybe add more feature map for conv layer may not improve the model slighty, because the number of feature map that model need, should based on complexity of the image, and the image of the training data seems simple, so I turn the hidden unit of conv layer to the original LeNet arch, and the performance did not change.
2.During the training,I found the training accurarcy is much higher than validation accurarcy, overfitting might happen.so I add dropout for each layer(include conv layer),now the accurarcy is about 94% and some times it even get 95%,but sometimes It got 92% or 93%. 
3.the training procedure seems unstable ,the accurarcy usually changed a lot for training and validation dataset.Why would this happen? Then I found that the channel of conv layer is very small, take the first layer for example, with keep_prob set to 0.6, the probability for more than 2 unit disabled can be very large, and less than 4 feature map is too small for feature extraction. so I canceled the dropout for conv layer, this time the model can often get 95%-96% accurarcy, and sometime even more (more than 96%).
* Which parameters were tuned? How were they adjusted and why?
My answer: I tried change the sigma of initial weight with 1 0.5 0.1(original),it seems both 1 or 0.5 are too large, 0.1 can get a very high accurarcy after 1 Epoch.
I think this happened because we have normalize the data and the varience of the weight should close to the varience of the data.  


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] 

The first and three images might be difficult to classify because it's very dark and the sign can't see clearly.image 1,4,5 can make the model confused easily,since they looks quiet similar.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit(80km/h)   | Speed Limit(80km/h)   						| 
| Turn left ahead     	| Turn left ahead   							|
| Keep right			| Keep right									|
| Speed Limit(30km/h)   | Speed Limit(30km/h)   						| 
| Speed Limit(60km/h)   | Speed Limit(60km/h)   						| 


![alt text][image4] 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. The model looks good!

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is very sure that this is a stop sign (probability of 99.9997377), and that is true. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 99.9997377         	| Speed Limit(80km/h)							    |    
| 2.21004666e-06     	| Speed Limit(100km/h)							    |
| 4.67203165e-07		| Speed Limit(60km/h)							    |
| 2.08533386e-08	    | Speed Limit(50km/h)			 				    |
| 4.59973881e-09		| End of no passing by vehicles over 3.5 metric tons|


![alt text][image4] 


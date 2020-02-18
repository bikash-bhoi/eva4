# Attempt 1
##  File summary

### Target

*   Build a model within 20K params, No AvgPool
* Use Dropout values : 0.1

### Result
* Param :  17.9K
* Best Train accuracy : 99.42 (20th)
* Best Test accuracy : 99.53 (15th)

### Analysis
* Regularizaition with lr=0.01 is doing a great job for ~18K parameters
* This is a great submission for previous assignment as the Test accuracy is consistently above 99.4 after 11th Epoch 


# Attempt 2
##  File summary

### Target

*   Lets be a miser and build a model with around 6K params params, Use AvgPool
* Reduce the #of channels

### Result
* Param :  5.6K
* Best Train accuracy : 98.92 (17th)
* Best Test accuracy : 99.17 (17th)

### Analysis

* Reducing the Numbers of params reduced the accuracies

# Attempt 3
##  File summary

### Target

*  Go one step deeper before avg pool 
*  Try LR Scheduler for MNIST. (**Nullius in verba**)

### Result
* Param :  8K
* Best Train accuracy : 99.24 (16th)
* Best Test accuracy : 99.40 (14th,15th)

### Analysis

* Going deeper helped, but we are still in fringe.

# Attempt 4
##  File summary

### Target

*  Use random rotation 

### Result
* Param :  8K
* Best Train accuracy : 99.04 (14th)
* Best Test accuracy : 99.45 (10th,15th)

### Analysis

* It helped the Test accuracy stabilized 

# Attempt 5
##  File summary

### Target

* Expand the model, Use as many as allowed parameters.
* Target reached but lets try and go extra mile.


### Result
* Param :  9.7K
* Best Train accuracy : 99.31 (12th)
* Best Test accuracy : 99.50 (12th)
* Earliest to cross target(Test accuracy) : 99.43(7th), 99.47(8th)

### Analysis
* Going deep helped reaching till 99.5
* Tweaking the initial LR, gamma and step size in LR_scheduler helped stabilizing the accuracy at the end

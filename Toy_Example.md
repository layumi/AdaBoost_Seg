## Will more “hard” samples to have a negative effect? 

Yes. We intend to obtain complementary snapshots as the final model and some single snapshots may overfit the negative sample, yielding a worse performance.
This phenomenon also occurs in the conventional Adaboost algorithm.

Given  x=[0,1,2,3,4,5,6], y = [1, -1,-1,-1, 1, 1,1], we train one binary classifier mapping x to y:  

In the first round, we will get the binary classifier. G1(x) = -1 (if x<3.5),  1 (if x≥3.5), only one sample is wrong.
The error rate is 1/7 = 14.3%. Then we update the data weights as [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] 
(Here just show the extreme case. Weight updating strategy can be different. ) 



In the second round, we will get a new binary classifier. G2(x) = -1 (if x<0),  1 (if x≥0)  
The error is 3/7 = 42.9%, which is worse than the error rate in the first round. 

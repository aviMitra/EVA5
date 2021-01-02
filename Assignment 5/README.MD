<h1>Assignment 5</h1>
Here we need to build Models in 4 steps of improvements so that the final model can consistently achieve an accuracy of 99.4% in MNIST Test Data. 
There were additional points for keeping the Parameters less than 8000. Each of the Colab Files in this folder contains the Target, results and their analysis. 
This doc will briefly recapitulate on the models.

<h2>Model 1</h2>
Initially a basic skeleton was created. The Goal was simple:
<ul>
<li> Create a basic structure that will be neatly reused for the othe rmodels
<li> Keep the no. of params less than 10k
<li> Use basic building blocks like COnv with Relu, Max-Pooling and also Use GAP in the end
<li> Achieve a baseline accuracy of 95%
</ul>

<h2>Model 2</h2>
The Goal was to improve upon Model 1 without changing the conv and Max-Pool structures:
<ul>
<li> Add batch-normaliztion since it makes the features representation by each kernel more clear and hence eaasier to detect.
<li> Add random Dropouts to each layer to make the job of training harder for the model and reduce overfitting
<li> Achieve a baseline accuracy of 99%
</ul>


<h2>Model 3</h2>
Further improve upon Model 3 without any structural or parameter changes:
<ul>
<li> Make the training job further difficult by introducing image rotation (-8 -> 8 fdegrees). This helps in reducing overfitting
<li> Achieve accuracy of 99.4% consitently in last 5 epochs
</ul>



<h2>Model 4</h2>
The target of the assignment is already achieved by Model 3. This model is to reduce the no. of params while keeping a similar accuracy:
<ul>
<li> Selectively Reduced the no. of Channels in COnv Layers
<li> Achieve accuracy of 99.4% consitently in last 5 epochs
</ul>


Download Link: https://assignmentchef.com/product/solved-eece-5644-homework-5
<br>
<span style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;">Make sure you read the problem statements and answer the questions. Submit the code online. This homework could be a bit challenging, so I suggest you start early.</span>

<strong>5.1 </strong>(50 pts) Bonus: Expectation Maximization for handwritten digit clustering. In this

problem, you will explore unsupervised learning with EM for a real application: hand­written digit recognition. The nice people at MIST have generated a dataset with many examples of handwritten digits (0-9). This dataset is provided as digitsSmall .mat with the data in a 3000 x 784 array. This array represents 3000 total samples of 28×28 (784) images of each digit. The label vector determines which digit corresponds to each

sample. To plot sample 5 from this array, you can use imagesc(reshape(inputDa.ta(5, ) ,28, )) ; The following figure shows the some example digits from this dataset.

<strong>5th sample                                      10th sample                                   33th sample</strong>




Figure 1: Example digits

You can also easily find the mean images by using the label comparison as logical meanDigit2 = mean(inputData(labels==2, )) ; . The following figure shows some mean images.

<strong>Mean for 2                                        Mean for 4                                         </strong><strong>Mean for 8</strong>




Figure 2: Example digits

We will perform clustering (unsupervised learning) using what you learned from EM; therefore, we won’t be using the labels provided. Instead of using Gaussian mixtures, we will use Bernoulli mixtures and treat the digits as binary images.

(a) (5 pts) In this problem, we are not using the labels provided in the dataset to learn the cluster dependent parameters. In your words, why would it be useful to develop




an unsupervised learning algorithm for this application? Can you think of other applications where labeling datasets is expensive?

(b) (10 pts) As mentioned before, we will treat is sample image as binary random variable modeled by a Bernoulli distribution (you can do this by thresholding the images with 0.5 i.e. inputDa.t a = inputData&gt;0 . 5 ; ). In other words, for a given x = <em>[a: 1 D], </em>we will have its probability given by

D

p(xia) = H tedd (1 — itd)’<sup>–</sup>wd

d=1

where <strong><em>p =           </em></strong><strong><em>• • • </em></strong><em>p <sub>D</sub>] </em>is the mean vector with the mean for each element. We are

treating each dimension as independent Bernoulli distributions. However, for the mixture model, we will treat each potential digit as its own Bernoulli. Therefore, the distribution for a sample x drawn from our dataset will be given by:

K

p(xibt, 7r) = E Thp(xitik)

k=1

where jc = {m<sub>u</sub> • • • , ,u <sub>K</sub>} (excuse my abuse of notation with <em>pt,) </em>and <strong><em>ir = </em></strong>[7r<sub>i</sub> • • • 71K<sup>–</sup>] are each of the component’s mean and prior, respectively, and

p(x           <em>= </em><strong>H </strong>P<sup>x</sup>kg<sup>1</sup> — <em>Pkd) </em>1 — <em>Xd</em>

<em>d=1</em>

From this set of equations, we can observe that the log-likelihood for a dataset X = {x<sub>1</sub>, x<sub>N</sub>} is given by




<table>

 <tbody>

  <tr>

   <td width="328">

    <table width="100%">

     <tbody>

      <tr>

       <td>

        <table>

         <tbody>

          <tr>

           <td width="148">ln p(X</td>

           <td width="30"><sup>7</sup>r)</td>

           <td width="85"><em>N K </em>=           Inn=i <em>k=1</em></td>

           <td width="50"><em><sup>71</sup></em><em>k1<sup>3</sup>( </em>xn</td>

           <td width="118">Pk)</td>

          </tr>

         </tbody>

        </table> </td>

      </tr>

     </tbody>

    </table></td>

  </tr>

 </tbody>

</table>




Like in the Gaussian mixture case, we can introduce the binary K-dimensional variable z = [z<sub>1</sub> • • z<sub>i</sub>d to indicate the component correspondence (only one ele­ment will be equal to 1 in this vector). Given this latent variable, we can write the distribution of x as:

K

p(x z,11) = 1173(xliak) k

<em>k=1</em>

<em>r)=</em>I7r<sup>k</sup> <em>= </em>1<sup>–</sup>1 <em>k=1</em>

With some work, it is possible to write the expectation of the complete-data log-likelihood as:

<em>N</em>

E<sub>z</sub> [ln p(X, Z <strong><em>IA, </em></strong><strong><em>Tr)] = </em></strong>E E <em>E[E„<sub>k</sub></em><em>] </em>liurr<sub>k</sub><em> L[1:</em><em><sub>7td</sub></em><em> In F </em><em>kd </em><em>+ (1 — </em><em>„</em><em><sub>d</sub></em><em>) </em>ln(1 —

<em>n=1 </em>k=1                            d=1




where Z = {z,} and X = {x<sub>n</sub>}. Front this, the E-step will be given by

<em>= </em><em><u>rkP(xnItik) </u></em><em> </em><em>Ek<sub>n</sub></em><em>ki  </em>

<em>El=1</em><em><sup>71-1</sup></em><em>P(<sup>x</sup>rb11</em><em><sup>1</sup></em><em>1)</em>

and the M-step by

<em>Nk = </em>E E[znk]

<em><u>Nk</u></em><em> 1</em>

<em><sub>N </sub></em><em>= <sub>N</sub></em><em><sup>—</sup></em> E E[znki

n=1

<em>N</em>

n=1

Prove the formula for <em>p</em><em><sub>q</sub></em> by setting the derivative of the E<sub>z</sub>[•] equation above to 0 ( derivative with respect to <em>P<sub>k</sub></em> of course).

(c) (10 pts) Implement the Expectation Maximization (EM) algorithm for Bernoulli clustering with the steps as shown in (b). The implementation MUST be a function that takes AT LEAST the following inputs:

<ol>

 <li>inputData: nSamples x dDimensions array with the data to be clustered</li>

 <li>numberOfClusters: number of cluster for algorithm</li>

</ol>

<ul>

 <li>stopTolerance: parameter for convergence criteria</li>

</ul>

<ol>

 <li>numberOfiluns: number of times the algorithm will run with random initial­izations</li>

</ol>

Notice that I say AT LEAST. You may want to add some optional parameters if you want to give the user more control over the algorithm. The function should output the results for best EM clustering. The output should be AT LEAST:

<ol>

 <li>clusterParameters: numberOfClusters x 1 struct array with the Bernoulli mixture parameters:</li>

</ol>

<ul>

 <li>.mu – mean of the Bernoulli component</li>

 <li>.prior – prior of the cluster component</li>

</ul>

Notice that I say AT LEAST. I suggest you also have the following outputs since they will make your plotting easier I think:

estimatedLabels: nSamples x 1 vector with labels based on maximum prob­ability. Since EM is a soft clustering algorithm, its output are just densities for each cluster in the mixture model

<ol>

 <li>logLikelihood: 1 x numberOfiterations vector with the log-likelihood as a function of iteration number</li>

</ol>




This algorithm is tricky to get right. I strongly suggest you run a simple tests firsts.

There are a couple of details you need to be mindful:

<ol>

 <li><strong>Initialization: </strong>your algorithm can end up in a local maximum. This is why one of the inputs is the number of times you will run EM with random By random initial conditions, <strong>I </strong>mean, select the <strong>initial K </strong><strong>mean vectors by drawing from a uniform distribution between 0.25 </strong><strong>and 0.75. </strong>You can make the initial priors to be uniform. Then, run your algorithm until convergence starting from each of these initial conditions. Finally, <strong>pick </strong>the one that results in the highest log-likelihood.</li>

 <li><strong>Convergence: </strong>from a practical viewpoint, you need to decide when it is no longer worth iterating. You could stop if the increase in log-likelihood between the last two iterations is less than 0.001% of the change in likelihood between the current value and the log-likelihood value after the very first iteration’ of the algorithm. For these data sets, you can also impose a maximum number of iterations to halt the algorithm (e.g., 50) if it gets that far and still has not converged.</li>

</ol>

<ul>

 <li><strong>Computational performance: </strong>computing E[z<sub>nk</sub>] can be tricky due to the size of the dataset. Think really hard bow to vectorize this operation. As a hint, you could use bsxfun or the exp(log(.)) trick to convert powers to</li>

 <li>(10 pts) <strong>Run </strong>the EM algorithm <strong>on digitsSma11.mat only for digits 0,1,4 </strong>with K=3. Make a 2×3 figure (with subplot) to generate the following plots: on the top <strong>row, </strong>the means for the clusters (they should look like a 0, <strong>1, </strong>and 4) and on the bottom row, the difference between these cluster means and the corresponding true means. The true mean for comparison can be computed as <strong>meanDigit0 = mean(inputDat a (labels==0 )&gt;0 .5) ; How </strong>do the cluster means compare<strong> it</strong> to the true ones?</li>

 <li>(15 pts) <strong>Run </strong>the EM algorithm on <strong>digit sSma.11 . mat </strong>for all digits for K=5,10,20. Make 3 figures (1×5,2×5,4×5) total, one for each K, with the images of the cluster means as found by your EM algorithm. Make sure you run enough initializations (50 for example). What can you say about the quality of the means as K increases? Did <strong>K=10 </strong>do a good job? What about K=20? What do you think could be the problem? Would a Gaussian mixture model improve these results? Would that be a good model for these images?</li>

</ul>

<strong>5.2 </strong>(50 pts) <strong>Support Vector Machines for handwritten digit classification. </strong>In this problem, you will use MATLAB’s SVM tool <strong>f itcsvm. </strong>Please, read the documentation carefully to learn its functionality. This experience will teach you how to use tools that you haven’t developed yourself. We will explore the multi-class extension to SVMs. We will use the smaller dataset to learn and use the bigger dataset to test the generalization capability of the classifier.




strengths and weaknesses of each approach

<ul>

 <li>(10 pts) For the next items, we will focus on the one-vs-all approach to multi-class SVM. I suggest you build a class for multiclass SVM and write Learn and Predict methods. Alternatively, you could write 2 functions: a “learn” one that outputs the set of SVM models that will be passed to the “predict” function. To learn the digits’ models, you will need to call <strong>fitcsvm 10 </strong>times with the appropriate positive arid negative labels. To predict, use the score that MATLAB’s <strong>predict </strong>function outputs for the 10 models and pick the label with the maximum score. Make sure you use the correct score in the output i.e. read the fun manual (RTFM).</li>

 <li>(10 pts) Use the <strong>mat </strong>to learn and <strong>digitsLarge .3:oat </strong>to test, Make a figure with an image of the confusion matrix (predicted labels vs true labels) with the total accuracy on the title. Use default options</li>

 <li>(10 pts) Use the <strong>mat </strong>to learn and <strong>digitsLarge.mat </strong>to test. Make a figure with an image of the confusion matrix (predicted labels vs true labels) with the total accuracy on the title. Use the radial basis function kernel for kernel-SVM with the kernel scale parameter as automatic (if not, you will perform poorly). Compare this to (c). Does RBF improve results? What does the automatic kernel scale do?</li>

 <li>(10 pts) Use the <strong>mat </strong>to learn and <strong>digitsLarge.mat </strong>to test. Make a figure with an image of the confusion matrix (predicted labels vs true labels) with the total accuracy on the title. Use the polynomial kernel with order 3. Compare this to (c)</li>

 <li>(10 pts) Bonus: explain why the RBF kernel implies an infinite dimensional feature space.</li>

</ul>
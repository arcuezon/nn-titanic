<h1>Kaggle Challenge for classifying Titanic survivors</h1>
<p>source: <a href = "https://www.kaggle.com/c/titanic/">https://www.kaggle.com/c/titanic/</a></p>

<p>This project is a simple DNN framework written from scratch to create an L-Layer DNN written in Python using numpy and pandas. The framework was used to complete the Titanic Kaggle Challenge.</p>

<p>This framework was written to try to apply what I have learned from Andrew Ng's and deeplearning.ai's Neural Networks course on Coursera. I realize that a Neural Net may not be the best or most efficient model to apply to the Titanic Problem but I just happen to come across this dataset first so I used it anyway. I did this to see if I could write an arbitrary-L layer NN from scratch (akin to a proof of concept of what I have learned in the course) and not really to be efficient or be the most robust network and certainly not rival the frameworks dominating the field</p>
<p> A lot can certainly be improved with the code but as I said, I really only wanted proof of concept of my understanding of Neural Nets.</p>

<h4>Best attempt:</h4>
<strong>Score of 0.779</strong>
<em>1-layer Neural Network with 3 units.</em>
<ul>
  <li>Activations</li>
  <ul>
    <li>ReLU for hidden layers</li>
    <li>Sigmoid for output layer</li>
  </ul>
  <li>Applied Learning rate decay</li>
  </ul>
  <em>Inupt Features</em>
  <ol>
    <li>Passenger Class</li>
    <li>Sex</li>
    <li>Age</li>
    <li># of Siblings or Spouses onboard</li>
    <li># of Parents or Children onboard</li>
    <li>Fare</li>
    <li>Port of Embarkation</li>
  </ol>

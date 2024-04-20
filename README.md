
<b><h2>Creating CNN model: </h2></b>

1. <h3>Input Layer:</h3>
   - The first layer of the CNN that receives the raw input data, usually an image or a set of images.</br>
   - It represents the input dimensions of your data, such as the height, width, and number of channels (e.g., RGB channels for color images).

2. <h3>Convolutional Layers:</h3>
   <b>- it detects the image pattern by using the sudden change in the pixel values and sudden decrease in the pixel values of a image </b></br>
   - These layers are the core building blocks of a CNN.</br>
   - Each convolutional layer consists of multiple filters or kernels that scan across the input image to detect patterns, edges, and features.</br>
   - Convolutional layers help the network automatically learn hierarchical representations of the input data.

3. <h3>Activation Function (ReLU):</h3>
   - After each convolutional operation, a non-linear activation function is applied, commonly the Rectified Linear Unit (ReLU).</br>
   - ReLU introduces non-linearity into the model, allowing it to learn complex patterns and relationships in the data.</br>

4. <h3>Pooling (Subsampling or Down-sampling) Layers:</h3>
   - Pooling layers reduce the spatial dimensions of the input data, helping to decrease computation in subsequent layers and make the network more robust to variations in input.</br>
   - Common pooling methods include max pooling and average pooling.

5. <h3>Flattening Layer:</h3>
   - The flattening layer is used to convert the multi-dimensional output of the previous layers into a one-dimensional vector.</br>
   - This is necessary to connect the convolutional and fully connected layers.

6. <h3>Fully Connected (Dense) Layers:</h3>
   - Dense layers connect every neuron in one layer to every neuron in the next layer.</br>
   - These layers are responsible for combining features learned by the previous layers to make predictions.</br>
  <b> - The number of neurons in the output layer is typically set based on the problem's requirements.</b>

7. <h3>Output Layer:</h3>
<b>-  this is the final  layer in the fully connected layer. based on the required out we need set no.of nuerons in this layer.</br>
</br> ---if prediction exists in </br>
      -----uncountable numbers then it is **LINEAR** </br>
      -----two things only then **BINARY**</br>
      -----countable but more than two then **MULTI_REGRESSIION**</br>
   - The final layer that produces the network's output.</br>
   - For binary classification problems (using binary cross-entropy), a single neuron with a sigmoid activation function is often used to produce a probability output between 0 and 1.

8. <h3>Activation Function (Sigmoid):</h3>
   - In the output layer, a sigmoid activation function is commonly used for binary classification tasks.</br>
   - It squashes the output values to the range [0, 1], representing the probability of belonging to the positive class.</br>

9. <h3>Loss Function (Binary Cross-Entropy):</h3>
   - Binary cross-entropy is the loss function used for binary classification problems.</br>
   - It measures the difference between the predicted probabilities and the actual class labels and is suitable for problems where each input belongs to exactly one of two classes.

10. <h3>Optimizer:</h3>
    - An optimizer is used to adjust the weights and biases of the network during training, minimizing the loss.</br>
    - Common optimizers include Adam, SGD (Stochastic Gradient Descent), and RMSprop.
</br>
        <h3>summary of the model:</h3>
     <img width="960" alt="Screenshot 2023-11-15 194159" src="https://github.com/harinadh14/CodeAlphaT1/assets/101285437/6fb06769-355d-49e5-84aa-1061c696e0cb">



11. <h3>Training:</h3>
<img width="960" alt="Screenshot 2023-11-15 194318" src="https://github.com/harinadh14/CodeAlphaT1/assets/101285437/a90eb6df-0f72-4525-ae21-0db4ca38df74">

12. <h3> <b>accuracy of the code :</b></h3>
<img width="920" alt="Screenshot 2023-11-15 194339" src="https://github.com/harinadh14/CodeAlphaT1/assets/101285437/54e92562-b79f-4eff-ab19-7fa99f55cf7f">

13.  <h3>MODEL-FIt</h3>
<img width="960" alt="Screenshot 2023-11-15 194318" src="https://github.com/harinadh14/CodeAlphaT1/assets/101285437/3ee97e44-0876-4d4d-b39e-4a166dc552ef">

14.  <h3>plotting the data using mat-plot lib : </h3>
<img width="960" alt="Screenshot 2023-11-15 194408" src="https://github.com/harinadh14/CodeAlphaT1/assets/101285437/43e2ab4a-b0d5-48e8-9434-26502f3b9c48">

    - The entire model is trained on a labeled dataset using backpropagation and gradient descent.</br>
    - The training process involves adjusting the model's weights based on the calculated loss, bringing the model closer to making accurate predictions.</br>
      <b><h3>this consists of the image generation. which generates the multiple different images in the run time to increase the accuracy of the code..</h3></b></br>
      <img width="873" alt="Screenshot 2023-11-15 194222" src="https://github.com/harinadh14/CodeAlphaT1/assets/101285437/342f8a9d-3939-4aa8-b99b-eb6991f99db4">

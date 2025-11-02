# Pushing-LeNet5-to-its-limits
This code tries to reproduce the original LeNet5 model, presented in Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791, and push it to its limits. The paper can also be found here chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf. The project is inspired by Andrej Karpathy's blog post https://karpathy.github.io/2022/03/14/lecun1989/ and his implementation of the 1989 LeNet model https://github.com/karpathy/lecun1989-repro

# Experimetns
The goal is to push the LeNet5 architecture to its limits. To do so, two experiments were constructed, using the MNIST dataset. In the first experiment the LeNet's architecture is the same as the original paper and the changes concern only the training process. Namely, Xavier initialization was used and the Adam optimizer with weight decay substituted SGD. Mini-batches of 32 samples were also used. In the second experiment, apart from the changes in the training phase, the model's architecture was updated. The average pooling layers were substituted by max pooling layers. Dropout was also applied after the flatten layer. ReLUs instead of tanh as activation functions and finally, batch normalization was added after the convolutional layers. In both experiments the output layer was kept as an radial basis function (RBF) layer and the loss function was MSE as in the original paper. The training, evaluation, and visualization processes were performed on an AMD Ryzen 9 5900HX CPU and an NVIDIA GeForce RTX 3070 Laptop GPU. 

# Results
Keeping the original architecture and changing only the training phase did not improve the model's performance, since the results were:
Number of misclassified examples: 104
Test Loss: 0.0025, Test Accuracy: 98.96%
Precision: 0.9896, Recall: 0.9896, F1 Score: 0.9896

Changing the architecture, helped a lot. Substituting tanh with ReLU, average pooling with max poolingm and adding dropout(0.25) before the flatten layer, gave:
Number of misclassified examples: 91
Test Loss: 0.0022, Test Accuracy: 99.09%
Precision: 0.9909, Recall: 0.9909, F1 Score: 0.9909

Reducing dropout to 0.1, improved further the results:
Number of misclassified examples: 83
Test Loss: 0.0020, Test Accuracy: 99.17%
Precision: 0.9917, Recall: 0.9917, F1 Score: 0.9917

Group normalization after the two convolutional layers boosted the model's performance even more:
Number of misclassified examples: 70
Test Loss: 0.0018, Test Accuracy: 99.30%
Precision: 0.9930, Recall: 0.9930, F1 Score: 0.9930

However, the change that gave the best performance was the incorporation of batch normalization after the two convolutional layers:
Number of misclassified examples: 66
Test Loss: 0.0017, Test Accuracy: 99.34%
Precision: 0.9934, Recall: 0.9934, F1 Score: 0.9934

In addition, the updated LeNet5 with batch norm converged in around ~60 epochs, in contrast to the others which needed around ~100 epochs.

# Conclusions
The LeNet5 architecture already used some of the "tricks" that modern CNN architectures use today. Batch normalization was the technique that helped the most, while Xavier initialization and Adam did not contribute much. The RBF output layer logic although does not scale as the linear logits and softmax layer and has been abandoned today, can be traced indirectly today in contrastive learning. 

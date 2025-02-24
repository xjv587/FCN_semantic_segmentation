# FCN_semantic_segmentation_DL_HW3
make this classification network fully convolutional and solve a semantic labeling task (labeling every pixel in the image).

# Tuning the classifier
Accuracy as high as possible by using:
- input normalization
- Residual blocks
- Dropout
- Data augmentations (Both geometric and color augmentations are important. Be aggressive here. Different levels of supertux have radically different lighting.)
- Weight regularization
- Early stopping
managed to tune our classifier to a  accuracy with a training time of 10 GPU minutes

# Dense prediction (semantic segmentation)
make the CNN fully convolutional (FCN). Instead of predicting a single output per image, you'll now predict an output per pixel.
Our current dataset does not support this output, we thus switch to a new dense prediction dataset, which can be found on Canvas. Here all images have a  resolution, and the labels are of the same size. We have 5 labels here: background, kart, track, bomb/projectile, pickup/nitro. We merged the last two classes, as they already have very few labeled instances. The class distribution for those labels is quite bad. background and track make up  of the labeled area! We will address this issue later for evaluation.

# FCN design
Design your FCN by writing the model in models.py. Make sure to use only convolutional operators, pad all of them correctly and match strided operators with up-convolutions. Use skip and residual connections.
Make sure your FCN handles an arbitrary input resolution and produces an output of the same shape as the input. Use output_padding=1 if needed. Crop the output if it is too large.

# FCN Training
To train your FCN you'll need to modify your CNN training code a bit. First, you need to use the DenseSuperTuxDataset. This dataset accepts a data augmentation parameters transform. Most standard data augmentation in torchvision do not directly apply to dense labeling tasks. We thus provide you with a smaller subset of useful augmentations that properly work with a pair of image and label in dense_transforms.py.
You will need to use the same bag of tricks as for classification to make the FCN train well.
Since the training set has a large class imbalance, it is easy to cheat in a pixel-wise accuracy metric. Predicting only track and background gives a  accuracy. We additionally measure the Intersection-over-Union evaluation metric. This is a standard semantic segmentation metric that penalizes largely imbalanced predictions. This metric is harder to cheat, as it computes  \frac{\text{true positives}}{\text{true positives} + \text{false positives} + \text{false negatives}}.  You might need to change the class weights of your torch.nn.CrossEntropyLoss, although our master solution did not require this. You can compute the IoU and accuracy using the ConfusionMatrix class.


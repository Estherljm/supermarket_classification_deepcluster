The objective of this project was to perform unsupervised classification on supermarket items using deepclustering technique

# Example of supermarket dataset images 
Source: https://www.kaggle.com/varunkashyapks/super-market-products
![alt text](https://github.com/Estherljm/supermarket_classification_deepcluster/blob/master/example.png)

# Workflow 
![alt text](https://github.com/Estherljm/supermarket_classification_deepcluster/blob/master/f1.png)
- ResNet50 and DarkNet53 were pretrained models on the ImageNet dataset 
- Using the features extracted, K-means algorithm is used to train the un-labeled dataset to get the centroid of the clusters
- The results are then clustered based on similarities given the initial input seeds, allowing for matching images to be pulled up and shared alongside, as the main purpose is to locate images with matching patterns based on the dataset of our choice

# Display of GUI 
![alt text](https://github.com/Estherljm/supermarket_classification_deepcluster/blob/master/gui.png)
- The image on the left showcases the inserted image by the user to get similar looking images 
- As we can see in the output images on the right, not all images showcased were correct
- The program was created to showcase the top 9 most similar objects/images to the input object/image 

# Results 
![alt text](https://github.com/Estherljm/supermarket_classification_deepcluster/blob/master/results.JPG)

- DarkNet53 performed best with 73.02% accuracy in displaying similar objects
- Out of all the test performed, all 3 models performed best when the shape of the object commonly conform to a distinct and similar design of packaging

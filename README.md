# Vehicle Detection Project - Peter Tempfli

Please see the final project output here:

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/-URKrWwJdz0/0.jpg)](https://www.youtube.com/watch?v=-URKrWwJdz0)


Please see the [project jupyter notebook](./project.ipynb) here.


## Histogram of Oriented Gradients (HOG)


### Hog feature extraction

For extracting the HOG features I'm using the default settings of skimage HOG algorithm on the L channel of the HLS color space.

![HOGS](./output_images/hogs.png)

```
def features_from_img(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1]
    vec = hog(img, visualise=False)
    return vec
```


### Classifier training

This peace of code shows how the training sets were built. I'm using the StandardScaler algorithm to normalize my training data. Once the data and the label sets are built, I'm generating a randomly sampled training and test set.

Here's a plot of one data point before and after normalization:

![Scaler](./output_images/scaler.png)

```
car_features = [features_from_img(img) for img in car_img]
non_car_features = [features_from_img(img) for img in non_car_img]
features = car_features + non_car_features

scaler = StandardScaler()
scaler.fit(features)

features = scaler.transform(features)
labels = [1 for i in range(len(car_features))] + [0 for i in range(len(non_car_features))]

X_train, X_test, y_train, y_test =  train_test_split(features, labels, test_size=0.15, random_state=999)
```
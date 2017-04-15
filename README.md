# Vehicle Detection Project - Peter Tempfli

Please see the final project output here:

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/URKrWwJdz0/0.jpg)](https://www.youtube.com/watch?v=-URKrWwJdz0)


Please see the [project jupyter notebook](./project.ipynb) here.


## Histogram of Oriented Gradients (HOG)

For extracting the HOG features I'm using the default settings of skimage HOG algorithm on the L channel of the HLS color space.

![HOGS](./output_images/hogs.png)

```
def features_from_img(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1]
    vec = hog(img, visualise=False)
    return vec
```



### Hog feature extraction



### Classifier training
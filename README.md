# pyCV
Python implementations of famous Computer Vision algorithms.

## SIFT: Scale-Invariant Feature Transform
Identifying stable features in images that are invariant to changes in object position, scale and luminosity.
A work in progress - currently, only feature extraction has been completed. 

```python
 SIFT_sample = SIFT()
 SIFT_sample.open_file('mike.jpg')
 SIFT_sample.detect()
```

### Note
These implementations served as learning experiences, and not to be the most efficient. The CV class also contains some often-used operations - convolutions, Guassian filters, derivative masks and such.
You might be interested in [OpenCV](https://github.com/opencv/opencv) for more performant implementations.

# PCA-machine-learning-reconstruct-teapot-image

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/f60b31ec-d3a3-42fc-bdfa-9a630d82a678)

I create a file call “problem1.m” I put my answer in there.
In this problem the question want us to implement PCA on teapot images to do 
image reconstruction, and we need to find top 3 most significant components of the
teapot image with the mean of the data to do image reconstruction.
The covariance matrix is then calculated, and the eigenvalues are determined. 
Given that the covariance matrix is both symmetric and positive-definite, all the 
eigenvalues are non-negative.
In this particular problem, we identify the three most significant eigenvalues, which 
are 4.2150, 3.0168, and 2.0993, each having a corresponding eigenvector.

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/cd53817a-9eae-4a7f-88f2-a0868039f5ba)

And the following figure 11 is the top 3 eigenvalues corresponding to [4.2150, 3.0168, 2.0993]:

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/6dbb9f94-b68c-417c-845e-2e4a0453eef9)

Next, a new coefficient matrix is computed using the first three components of the 
eigenvectors, and the images are reconstructed accordingly. Subsequently, we 
present visual representations of 10 images, both before and after the reconstruction 
process. From figure 1 ~ figure 10 are the 10 before after reconstruction images.

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/815df28c-976c-45a4-857b-30765af49e7a)

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/8350da9f-5382-464f-a230-449782de7370)

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/70a8bd93-89bf-4c63-862d-8e8b0a182682)

![image](https://github.com/wayne540500/PCA-machine-learning-reconstruct-teapot-image/assets/69573286/2a7a3eab-a63c-4ff9-8361-47f15035b2d4)

In our analysis, we've observed that the reconstruction of teapot images using PCA 
varies in quality. Some images are well-reconstructed, while others might be 
considered to have a lower quality of reconstruction. It's worth noting that a more 
comprehensive reconstruction can likely be achieved by decoding with a larger 
number of components.
To evaluate the performance of the PCA encoder, we computed the L2-norm. With 
the top-3 components, the norm is 13.6262. However, as we increase the number of 
components used in reconstruction, the norm reduces, indicating improved results. 
This observation suggests that including more components in the reconstruction 
process leads to a better outcome, with reduced reconstruction error.
The observations from this analysis also suggest that a larger number of principal 
components leads to better image reconstruction. This insight is valuable when 
deciding on the number of components to use for dimensionality reduction and 
image compression in practical applications

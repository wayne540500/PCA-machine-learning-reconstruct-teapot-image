% Load the teapot dataset
load('teapots.mat');
teapotImages = teapotImages;  

% Calculate the mean and de-center the data
data_mean = mean(teapotImages);
decentered_data = teapotImages - data_mean;

% Compute the covariance matrix
cov_matrix = cov(decentered_data);

% Calculate the top 3 eigenvalues and eigenvectors
[V, D] = eig(cov_matrix);
eigenvalues = diag(D);
[sorted_eigenvalues, indices] = sort(eigenvalues, 'descend');
top_3_eigenvalues = sorted_eigenvalues(1:3);
top_3_eigenvectors = V(:, indices(1:3));

% Display the top 3 eigenvalues
disp('Top 3 Eigenvalues:');
disp(top_3_eigenvalues);

figure (11);
for i = 1:3
    subplot(1, 3, i);
    eigenvalue_image = reshape(V(:, indices(i)), 38, 50);
    imagesc(eigenvalue_image);
    colormap gray;
    title(['Eigenvalue ', num2str(i)]);
    axis image;
end

% Reconstruct the data using PCA with the top 3 eigenvectors
coefficients = decentered_data * top_3_eigenvectors;
reconstructed_data = data_mean + coefficients * top_3_eigenvectors';

% Calculate and display the Frobenius norm to measure reconstruction error
reconstruction_error = norm(teapotImages - reconstructed_data);
disp('Frobenius Norm (Reconstruction Error):');
disp(reconstruction_error);

% Plot 10 different images before and after reconstruction
num_samples = 10;
for i = 1:num_samples
    figure(i);
    
    % Before reconstruction
    subplot(1, 2, 1);
    imagesc(reshape(teapotImages(i, :), 38, 50));
    colormap gray;
    title('Before reconstruction');
    axis image;
    
    % After reconstruction
    subplot(1, 2, 2);
    imagesc(reshape(reconstructed_data(i, :), 38, 50));
    colormap gray;
    title('After reconstruction');
    axis image;
end

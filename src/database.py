import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Database:
    """
    Database class for VPR, storing features and performing searches
    """
    def __init__(self, vpr, method='mse', threshold=0.25):
        self.vpr = vpr
        self.method = method
        self.THRESHOLD = threshold
        self.database = {}
        if self.method == 'knn':
            self.knn_model = NearestNeighbors(n_neighbors=1, algorithm='auto')

    def _check_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Oops! The directory or file {path} does not exist.")

    def populate(self, sequence_dir, batch_size=32):
        self._check_path(sequence_dir)
        all_image_paths = [os.path.join(root, filename)
                       for root, _, files in os.walk(sequence_dir)
                       for filename in files
                       if filename.endswith(('.png', '.jpg'))]
        for i in range(0, len(all_image_paths), batch_size):
            batch_paths = all_image_paths[i:i+batch_size]
            batch_features = self.vpr.extract_features_batch(batch_paths)
            for path, feature in zip(batch_paths, batch_features):
                self.database[path] = feature

        if self.method == 'knn':
            features_array = np.array([feat.numpy() for feat in self.database.values()])
            self.knn_model.fit(features_array)

    def get_best_match(self, image_path, search_method='mse'):
        self._check_path(image_path)
        test_features = self.vpr.extract_features_batch([image_path])[0]
        if search_method == 'knn':
            distances, indices = self.knn_model.kneighbors([test_features.numpy()])
            if distances[0][0] < self.THRESHOLD:
                best_match_path = list(self.database.keys())[indices[0][0]]
                return best_match_path
            else:
                return "No match found"
        else:
            # Default to MSE-based matching
            min_distance = float('inf')
            best_match_path = None
            for path, features in self.database.items():
                distance = torch.nn.functional.mse_loss(features, test_features).item()
                if distance < min_distance:
                    min_distance = distance
                    best_match_path = path
            if min_distance > self.THRESHOLD:
                return "No match found"
            else:
                return best_match_path

    def get_all_matches(self, image_path, search_method='mse', n_best=5):
        """
        Get up to n best matches for a given image based on either KNN or MSE.

        Returns:
        - matches: list of str, the paths of the matching images
        """
        test_features = self.vpr.extract_features_batch([image_path])[0]
        matches = []

        if search_method == 'knn':
            distances, indices = self.knn_model.kneighbors([test_features.numpy()])
            close_indices = np.where(distances[0] <= self.THRESHOLD)[0]
            matches = [list(self.database.keys())[i] for i in indices[0][close_indices]]

        else:  # MSE-based matching
            database_keys = [path for path in self.database.keys() if path != image_path]
            database_features = [feat for path, feat in self.database.items() if path != image_path]
            all_features = torch.stack(database_features)
            test_features_expanded = test_features.unsqueeze(0)
            mse_distances = ((all_features - test_features_expanded) ** 2).mean(-1)
            mse_distances_np = mse_distances.detach().cpu().numpy()

            close_indices = np.where(mse_distances_np <= self.THRESHOLD)[0]
            close_keys = [database_keys[i] for i in close_indices]
            matches = close_keys

        # Return up to N matches
        return matches[:n_best]
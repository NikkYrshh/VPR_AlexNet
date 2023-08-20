from vpr_alexnet_class import VPRClass
from database import Database
from test_sequence import test_sequence

# Main
if __name__ == '__main__':
    vpr = VPRClass()

    method = input("Choose a method ('mse' or 'knn'): ").strip().lower()
    # method = ""  # choose method 'knn' or 'mse'

    test_sequence_dir = "/images"  # Adjust with your directory

    if method == 'mse':
        db = Database(vpr, method='mse')  # MSE
        print("Populating database with MSE method...")
        db.populate(test_sequence_dir)

        print("Testing with MSE method...")
        _, mse_match_counter, mse_time = test_sequence(test_sequence_dir, db)
        print(f"MSE-based matching: Total matches = {mse_match_counter}, Time = {mse_time:.2f} seconds")

    elif method == 'knn':
        db = Database(vpr, method='knn')  # KNN
        print("Populating database with KNN method...")
        db.populate(test_sequence_dir)

        print("Testing with KNN method...")
        _, knn_match_counter, knn_time = test_sequence(test_sequence_dir, db)
        print(f"KNN-based matching: Total matches = {knn_match_counter}, Time = {knn_time:.2f} seconds")

    else:
        print("Invalid method. Please choose 'mse' or 'knn'")

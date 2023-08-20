import os
import time
import matplotlib.pyplot as plt
from PIL import Image
def test_sequence(sequence_dir, db, n_best=5):
    """
    Test the database with a sequence of images

    Returns:
    - results: dict, the best matches for each test image
    - match_counter: int, the number of test images that found a match
    - elapsed_time: float, the elapsed time for the test sequence
    """
    match_counter = 0
    total_queries = 0
    results = {}

    all_images = sorted(
        [os.path.join(sequence_dir, fname) for fname in os.listdir(sequence_dir) if fname.endswith(('.png', '.jpg'))]
    )

    start_time = time.time()

    for image_path in all_images:
        total_queries += 1
        matches = db.get_all_matches(image_path, n_best=n_best)

        if matches:
            match_counter += 1

            # Visualization code, now modified to display only n_best matches
            fig, axes = plt.subplots(1, len(matches) + 1, figsize=(5 * (len(matches) + 1), 5))
            test_img = Image.open(image_path)
            axes[0].imshow(test_img)
            axes[0].set_title("Test Image")
            axes[0].axis("off")

            for i, match_path in enumerate(matches):
                match_img = Image.open(match_path)
                axes[i + 1].imshow(match_img)
                axes[i + 1].set_title(f"Match {i + 1}")
                axes[i + 1].axis("off")

            plt.show()
            results[image_path] = matches

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total queries: {total_queries}")
    print(f"Total matches: {match_counter}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    return results, match_counter, elapsed_time
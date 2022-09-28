import unittest
from pathlib import Path
import imagerec.tests.images as test_images
from imagerec.helpers import get_path_to, get_image_from
from imagerec.infer import infer

class TestInfer(unittest.TestCase):
    def test_infer_all_test_images(self):
        test_images_folder = get_path_to(test_images)
        image_labels_match = {
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
        }

        test_images_files = test_images_folder.glob("*.jpg")
        for test_image_file in test_images_files:
            actual_label = test_image_file.stem
            # convert the number to the label
            if image_labels_match.get(actual_label):
                actual_label = image_labels_match[actual_label]
            image = get_image_from(test_image_file)

            predicted_label = infer(image)[0]

            self.assertEqual(actual_label, predicted_label)

if __name__ == "__main__":
    unittest.main()

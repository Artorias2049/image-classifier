import os
import shutil

from cleanvision import Imagelab


def clean_problematic_directories(dataset_path):
    """Remove directories with image extensions"""
    bad_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for item in dirs:
            if item.endswith((".jpg", ".jpeg", ".png")):
                bad_path = os.path.join(root, item)
                bad_paths.append(bad_path)
                print(
                    f"Warning: Found directory with image extension: {bad_path}"
                )

    if bad_paths:
        print(f"Removing {len(bad_paths)} directories with image extensions")
        for path in bad_paths:
            try:
                os.rmdir(path)  # Only removes if empty
                print(f"Removed empty directory: {path}")
            except OSError:
                try:
                    # Try to remove with all contents if not empty
                    shutil.rmtree(path)
                    print(f"Removed directory and contents: {path}")
                except Exception as e2:
                    print(f"Could not remove directory {path}: {e2}")
    return len(bad_paths)


def clean_dataset(dataset_path, output_dir):
    """Clean dataset using CleanVision and remove problematic images"""
    print(f"Running CleanVision on {dataset_path}...")

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist")
        return

    # First remove problematic directories
    clean_problematic_directories(dataset_path)

    image_count = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_count += 1

    if image_count == 0:
        print(f"No image files found in {dataset_path}")
        return

    print(f"Found {image_count} image files to analyze.")

    try:
        os.makedirs(output_dir, exist_ok=True)

        lab = Imagelab(dataset_path)
        lab.find_issues()

        # Generate report
        try:
            lab.report()
        except Exception as e:
            print(f"Warning: Error generating report: {e}")

        print(f"Issues detected in {dataset_path}:")
        print(lab.issue_summary)

        # Direct access to the 'issues' DataFrame
        if hasattr(lab, "issues"):
            # Dictionary mapping issue column names to readable names
            issue_columns = {
                "is_dark_issue": "dark",
                "is_light_issue": "light",
                "is_low_information_issue": "low information",
                "is_odd_aspect_ratio_issue": "odd aspect ratio",
                "is_blurry_issue": "blurry",
                "is_grayscale_issue": "grayscale",
                "is_odd_size_issue": "odd size",
                "is_exact_duplicates_issue": "exact duplicate",
                "is_near_duplicates_issue": "near duplicate",
            }

            # Access the issues that should be removed
            problem_images = []
            issue_counts = {}

            for col, issue_name in issue_columns.items():
                if col in lab.issues.columns:
                    # Get indices of images with this issue
                    issue_images = lab.issues.index[
                        lab.issues[col] is True
                    ].tolist()
                    problem_images.extend(issue_images)

                    # Count for reporting
                    if issue_images:
                        issue_counts[issue_name] = len(issue_images)

            # Remove duplicates by converting to set and back to list
            problem_images = list(set(problem_images))

            # Print summary
            print("\nProblematic images by issue type:")
            for issue, count in issue_counts.items():
                print(f"  {issue}: {count}")

            print(
                f"\nFound {len(problem_images)} unique problematic images to remove"
            )

            # Remove the images
            removed = 0
            for img_path in problem_images:
                full_path = os.path.join(dataset_path, img_path)
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    try:
                        os.remove(full_path)
                        removed += 1
                        # Print a few examples
                        if removed <= 5:
                            print(f"Removed: {img_path}")
                        elif removed % 20 == 0:
                            print(f"Removed {removed} images so far...")
                    except Exception as e:
                        print(f"Error removing {img_path}: {e}")

            print(f"Successfully removed {removed} problematic images")
        else:
            print("No 'issues' attribute found in CleanVision object")

    except Exception as e:
        print(f"Error in CleanVision processing: {e}")
        print("Continuing without cleaning...")

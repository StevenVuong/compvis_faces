import pandas as pd
import subprocess
import os

output_dir = "/Users/steven/Downloads/semseg_vid_extract/input"
batch_base_dir = "/Users/steven/Downloads/semseg_vid_extract/"

def batch_csv_to_df(csv_path):

    batch_df = pd.read_csv(
        csv_path,
        names = ["gcs_url", "num_frames", "annotation_ref","upload_date"],
        skiprows=1,
        )
    batch_df = batch_df.drop(columns=["upload_date"])
    batch_df = batch_df[batch_df["num_frames"] != 100]

    return batch_df


def download_extract_clear(batch_df, output_dir, batch_base_dir):

    for index,row in batch_df.iterrows():

        gcs_url = row["gcs_url"]
        num_frame = row["num_frames"]
        annotation_ref = row["annotation_ref"]

        # Download
        subprocess.run(
            [r"gsutil", "cp", gcs_url, output_dir], check=True
            )

        # if not exist, make annotation ref folder and copy images
        batch_save_path = batch_base_dir + annotation_ref + "/"
        if not os.path.exists(batch_save_path):
            os.makedirs(batch_save_path)

        # Extract Frames
        subprocess.run(
            [r"python3",
             "./extract_frame_sample_image.py",
             str(num_frame),
             batch_save_path]
        )

        # Clear input
        for vid_file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, vid_file))


if __name__=="__main__":

    for i in range(7, 8): # test this loop

        csv_path = f"/Users/steven/Downloads/Manual Segmentation Scan Upload Tracker - Batch {i}.csv"
        
        batch_df = batch_csv_to_df(csv_path)
        download_extract_clear(batch_df, output_dir, batch_base_dir)
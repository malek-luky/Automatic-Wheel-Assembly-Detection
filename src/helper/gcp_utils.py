import io

import pandas as pd
from google.cloud import secretmanager, storage


def load_bucket_data(bucket_name, path_in_bucket):
    """
    Load data from a GCS bucket into a Pandas DataFrame.

    :param bucket_name: Name of the GCS bucket.
    :param path_in_bucket: Path to the file in the bucket.
    :return: Pandas DataFrame.
    """

    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob (file)
    blob = bucket.blob(path_in_bucket)

    # Download the contents as a bytes object
    data = blob.download_as_bytes()

    # Convert bytes to a string buffer and then to a DataFrame
    return pd.read_csv(io.StringIO(data.decode("utf-8")))


def get_secret(project_id, secret_id, version_id="latest"):
    """
    Retrieve a secret from Google Cloud Secret Manager.

    :param project_id: Google Cloud project ID.
    :param secret_id: ID of the secret in the Secret Manager.
    :param version_id: Version of the secret; defaults to 'latest'.
    :return: The secret as a string.
    """

    # Initialize the Secret Manager client
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version
    response = client.access_secret_version(request={"name": name})

    # Return the secret payload as a string
    return response.payload.data.decode("UTF-8")


# Usage
if __name__ == "__main__":
    data = load_bucket_data("wheel-assembly-detection-dataset", "data/processed/dataset_concatenated.csv")
    secret = get_secret("wheel-assembly-detection", "WANDB_API_KEY")
    print(secret)

import json
import boto3
import logging
from typing import Optional

from fooocusapi.utils.logger import logger

def upload_to_s3(s3_client, bucket: str, key: str, file_path: str):
    """Upload a file to S3"""
    try:
        s3_client.upload_file(file_path, bucket, key)
        logger.std_info(f"[S3] Uploaded {key} to bucket {bucket}")
    except Exception as e:
        logger.std_error(f"[S3] Failed to upload {key} to bucket {bucket}: {e}")
        raise e

def download_from_s3(s3_client, key: str) -> Optional[str]:
    """Download a file from S3 and return the local file path"""
    local_path = f"/tmp/{key.split('/')[-1]}"
    try:
        s3_client.download_file(key.split('/')[0], '/'.join(key.split('/')[1:]), local_path)
        logger.std_info(f"[S3] Downloaded {key} to {local_path}")
        return local_path
    except Exception as e:
        logger.std_error(f"[S3] Failed to download {key} from S3: {e}")
        return None

def send_sqs_message(sqs_client, queue_url: str, message_body: dict):
    """Send a message to an SQS queue"""
    try:
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message_body)
        )
        logger.std_info(f"[SQS] Sent message to {queue_url}, MessageId: {response.get('MessageId')}")
    except Exception as e:
        logger.std_error(f"[SQS] Failed to send message to {queue_url}: {e}")
        raise e
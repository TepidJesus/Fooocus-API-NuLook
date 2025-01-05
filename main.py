# -*- coding: utf-8 -*-

""" Entry for Fooocus API with SQS integration.

Use for starting Fooocus API.
    python main.py --help for more usage

@file: main.py
@author: Konie
@update: 2024-03-22 
"""
import argparse
import os
import re
import shutil
import sys
import json
from threading import Thread
import datetime

import boto3  # AWS SDK for Python
from fooocusapi.models.common.base import Lora
from fooocusapi.utils.logger import logger
from fooocusapi.utils.tools import run_pip, check_torch_cuda, requirements_check
from fooocusapi.base_args import add_base_args

from fooocus_api_version import version

script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(script_path, "repositories/Fooocus")

sys.path.append(script_path)
sys.path.append(module_path)

logger.std_info("[System ARGV] " + str(sys.argv))

try:
    index = sys.argv.index('--gpu-device-id')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[index+1])
    logger.std_info(f"[Fooocus] Set device to: {str(sys.argv[index+1])}")
except ValueError:
    pass

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

python = sys.executable
default_command_live = True
index_url = os.environ.get("INDEX_URL", "")
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def download_models_from_s3():
    """
    Download model and LoRA files from S3 and place them in the correct local directories.
    """
    s3_bucket_name = os.environ.get('MODEL_BUCKET_NAME', 'nulook-prod-models')
    aws_region = os.environ.get('AWS_REGION', 'us-east-2')

    if not s3_bucket_name:
        logger.std_error("MODEL_BUCKET_NAME environment variable not set.")
        return

    s3_client = boto3.client('s3', region_name=aws_region)

    # Define the local paths
    model_local_path = os.path.join(script_path, 'repositories', 'Fooocus', 'models', 'checkpoints')
    lora_local_path = os.path.join(script_path, 'repositories', 'Fooocus', 'models', 'loras')
    prompt_expansion_path = os.path.join(script_path, 'repositories', 'Fooocus', 'models', 'prompt_expansion', 'fooocus_expansion')
    
    # Ensure all directories exist
    os.makedirs(model_local_path, exist_ok=True)
    os.makedirs(lora_local_path, exist_ok=True)
    os.makedirs(prompt_expansion_path, exist_ok=True)
    
    # Download prompt expansion model
    prompt_expansion_s3_key = '/prompt_expansion/pytorch_model.bin'  # Adjust this path as needed
    try:
        logger.std_info("Downloading prompt expansion model from S3")
        s3_client.download_file(
            s3_bucket_name,
            prompt_expansion_s3_key,
            os.path.join(prompt_expansion_path, 'pytorch_model.bin')
        )
        logger.std_info(f"Downloaded prompt expansion model from S3 bucket {s3_bucket_name}")
    except Exception as e:
        logger.std_error(f"Error downloading prompt expansion model: {e}")

    flax_s3_key = '/prompt_expansion/flax_model.msgpack'  # Adjust this path as needed
    try:
        logger.std_info("Downloading flax model from S3")
        s3_client.download_file(
            s3_bucket_name,
            flax_s3_key,
            os.path.join(prompt_expansion_path, 'flax_model.msgpack')
        )
        logger.std_info(f"Downloaded flax model from S3 bucket {s3_bucket_name}")
    except Exception as e:
        logger.std_error(f"Error downloading flax model: {e}")

    ckpt_s3_key = '/prompt_expansion/model.ckpt.index'  # Adjust this path as needed
    try:
        logger.std_info("Downloading model.ckpt.index model from S3")
        s3_client.download_file(
            s3_bucket_name,
            ckpt_s3_key,
            os.path.join(prompt_expansion_path, 'model.ckpt.index')
        )
        logger.std_info(f"Downloaded model.ckpt.index model from S3 bucket {s3_bucket_name}")
    except Exception as e:
        logger.std_error(f"Error downloading model.ckpt.index model: {e}")

        tf_model_s3_key = '/prompt_expansion/tf_model.h5'  # Adjust this path as needed
    try:
        logger.std_info("Downloading tf_model from S3")
        s3_client.download_file(
            s3_bucket_name,
            tf_model_s3_key,
            os.path.join(prompt_expansion_path, 'tf_model.h5')
        )
        logger.std_info(f"Downloaded tf_model from S3 bucket {s3_bucket_name}")
    except Exception as e:
        logger.std_error(f"Error downloading tf_model.h5: {e}")
        
    # Download the model file
    model_s3_key = 'models/juggernautXL_v8Rundiffusion.safetensors'
    try:
        logger.std_info("Downloading model from S3")
        s3_client.download_file(
            s3_bucket_name,
            model_s3_key,
            os.path.join(model_local_path, 'juggernautXL_v8Rundiffusion.safetensors')
        )
        logger.std_info(f"Downloaded model {model_s3_key} from S3 bucket {s3_bucket_name}")
    except Exception as e:
        logger.std_error(f"Error downloading model file from S3: {e}")

    # Download all LoRA files from the /loras folder
    loras_s3_prefix = 'loras/'
    try:
        logger.std_info("Downloading loras from S3")
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=loras_s3_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue  # Skip directories
                local_file_name = os.path.basename(key)
                local_file_path = os.path.join(lora_local_path, local_file_name)
                s3_client.download_file(s3_bucket_name, key, local_file_path)
                logger.std_info(f"Downloaded LoRA {key} from S3 bucket {s3_bucket_name}")
    except Exception as e:
        logger.std_error(f"Error downloading LoRA files from S3: {e}")


def install_dependents(skip: bool = False):
    """
    Check and install dependencies
    Args:
        skip: skip pip install
    """
    if skip:
        return

    torch_index_url = os.environ.get(
        "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121"
    )

    # Check if you need pip install
    if not requirements_check():
        run_pip("install -r requirements.txt", "requirements")

    if not check_torch_cuda():
        run_pip(
            f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}",
            desc="torch",
        )


def preload_pipeline():
    """Preload pipeline"""
    logger.std_info("[Fooocus-API] Preloading pipeline ...")
    import modules.default_pipeline as _


def prepare_environments(args) -> bool:
    """
    Prepare environments
    Args:
        args: command line arguments
    """

    if args.base_url is None or len(args.base_url.strip()) == 0:
        host = args.host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        args.base_url = f"http://{host}:{args.port}"

    sys.argv = [sys.argv[0]]

    # Remove and copy preset folder
    origin_preset_folder = os.path.abspath(os.path.join(module_path, "presets"))
    preset_folder = os.path.abspath(os.path.join(script_path, "presets"))
    if os.path.exists(preset_folder):
        shutil.rmtree(preset_folder)
    shutil.copytree(origin_preset_folder, preset_folder)

    from modules import config
    from fooocusapi.configs import default

    default.default_inpaint_engine_version = config.default_inpaint_engine_version
    default.default_styles = config.default_styles
    default.default_base_model_name = config.default_base_model_name
    default.default_refiner_model_name = config.default_refiner_model_name
    default.default_refiner_switch = config.default_refiner_switch
    default.default_loras = config.default_loras
    default.default_cfg_scale = config.default_cfg_scale
    default.default_prompt_negative = config.default_prompt_negative
    default.default_aspect_ratio = default.get_aspect_ratio_value(
        config.default_aspect_ratio
    )
    default.available_aspect_ratios = [
        default.get_aspect_ratio_value(a) for a in config.available_aspect_ratios
    ]

    # Download models from S3
    download_models_from_s3()

    # Init task queue
    from fooocusapi import worker
    from fooocusapi.task_queue import TaskQueue

    worker.worker_queue = TaskQueue(
        queue_size=args.queue_size,
        history_size=args.queue_history,
        webhook_url=args.webhook_url,
        persistent=args.persistent,
    )

    logger.std_info(f"[Fooocus-API] Task queue size: {args.queue_size}")
    logger.std_info(f"[Fooocus-API] Queue history size: {args.queue_history}")
    logger.std_info(f"[Fooocus-API] Webhook url: {args.webhook_url}")

    return True


def pre_setup():
    """
    Pre setup, for replicate
    """
    class Args(object):
        """
        Arguments object
        """
        host = "127.0.0.1"
        port = 8888
        base_url = None
        sync_repo = "skip"
        disable_image_log = True
        skip_pip = True
        preload_pipeline = True
        queue_size = 100
        queue_history = 0
        preset = "default"
        webhook_url = None
        persistent = False
        always_gpu = False
        all_in_fp16 = False
        gpu_device_id = None
        apikey = None

    print("[Pre Setup] Prepare environments")

    arguments = Args()
    sys.argv = [sys.argv[0]]
    sys.argv.append("--disable-image-log")

    install_dependents(arguments.skip_pip)

    prepare_environments(arguments)

    # Start task schedule thread
    from fooocusapi.worker import task_schedule_loop

    task_thread = Thread(target=task_schedule_loop, daemon=True)
    task_thread.start()

    print("[Pre Setup] Finished")


import boto3
import base64
from io import BytesIO

def sqs_polling_loop():
    """
    Poll the SQS queue for messages and process them.
    """
    import base64
    aws_region = os.environ.get('AWS_REGION', 'us-east-2')
    inbound_queue_url = os.environ.get('ML_ENGINE_QUEUE_URL', 'https://sqs.us-east-2.amazonaws.com/058264145885/NuLookDevImageMLEngineThomas')
    outbound_queue_url = os.environ.get('POST_PROCESSING_QUEUE_URL', 'https://sqs.us-east-2.amazonaws.com/058264145885/NuLookDevImagePostProcessingThomas')
    s3_bucket_name = os.environ.get('PRE_PROCESSED_IMAGE_BUCKET_NAME', 'nulook-dev-preprocessed-images')
    s3_result_bucket_name = os.environ.get('INPAINTED_IMAGE_BUCKET_NAME', 'nulook-dev-inpainted-images')
    dynamodb_table_name = os.environ.get('JOB_STATUS_TABLE', 'NuLookJobStatus')

    s3_client = boto3.client('s3', region_name=aws_region)
    sqs_client = boto3.client('sqs', region_name=aws_region)
    dynamodb_client = boto3.client('dynamodb', region_name=aws_region)

    logger.std_info("[SQS Polling] Started polling for messages.")

    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=inbound_queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10
            )

            messages = response.get('Messages', [])
            if not messages:
                continue

            for message in messages:
                receipt_handle = message['ReceiptHandle']
                body = message['Body']
                msg = json.loads(body)
                job_id = msg.get('job_id', None)
                logger.std_info(f"[SQS Message] Received message: {msg}")

                # Record the time when processing starts
                start_time = datetime.datetime.utcnow().isoformat()

                # Update DynamoDB to reflect that inpainting has started
                dynamodb_client.update_item(
                    TableName=dynamodb_table_name,
                    Key={
                        'job_id': {'S': job_id}
                    },
                    UpdateExpression='SET job_status = :status, updated_at = :updated_at',
                    ExpressionAttributeValues={
                        ':status': {'S': 'INPAINT_STARTED'},
                        ':updated_at': {'S': start_time}
                    }
                )

                # Extract parameters from the message
                params = msg.copy()
                input_image_key = params.pop('input_image_key', None)
                input_mask_key = params.pop('input_mask_key', None)
                job_id = params.pop('job_id', None)

                # Download and base64-encode input image from S3
                if input_image_key:
                    response = s3_client.get_object(Bucket=s3_bucket_name, Key=input_image_key)
                    input_image_data = response['Body'].read()
                    # Since the image is stored as a base64-encoded string, decode it to get the string
                    input_image_base64 = input_image_data.decode('utf-8')
                else:
                    logger.std_error("[SQS Message] No input_image_key provided.")
                    continue

                # Download input mask from S3 (if provided)
                input_mask_base64 = None
                if input_mask_key:
                    response = s3_client.get_object(Bucket=s3_bucket_name, Key=input_mask_key)
                    input_mask_data = response['Body'].read()
                    input_mask_base64 = input_mask_data.decode('utf-8')

                # Set default values for parameters not provided
                from fooocusapi.models.requests_v2 import ImgInpaintOrOutpaintRequestJson, ImagePrompt
                from fooocusapi.models.common.requests import AdvancedParams
                from fooocusapi.routes.generate_v2 import call_worker

                # Create default advanced parameters
                advanced_params_obj = AdvancedParams(inpaint_erode_or_dilate=params.get('inpaint_erode_or_dilate', 15), 
                                                     inpaint_respective_field=params.get('inpaint_respective_field', 0.75), 
                                                     inpaint_strength=params.get('inpaint_strength', 1.0),
                                                     inpaint_disable_initial_latent=params.get('inpaint_disable_initial_latent', False),
                )

                # Create default image prompts
                image_prompts = params.get('image_prompts', [])
                image_prompts_files = []
                for image_prompt in image_prompts:
                    cn_img_key = image_prompt.get('cn_img_key')
                    if cn_img_key:
                        # Download and base64-encode controlnet image from S3
                        response = s3_client.get_object(Bucket=s3_bucket_name, Key=cn_img_key)
                        cn_img_data = response['Body'].read()
                        cn_img_base64 = base64.b64encode(cn_img_data).decode('utf-8')
                    else:
                        cn_img_base64 = None
                    image_prompt_obj = ImagePrompt(
                        cn_img=cn_img_base64,
                        cn_stop=image_prompt.get('cn_stop', 0.0),
                        cn_weight=image_prompt.get('cn_weight', 1.0),
                        cn_type=image_prompt.get('cn_type', 'None')
                    )
                    image_prompts_files.append(image_prompt_obj)

                # Ensure minimum number of image_prompts
                while len(image_prompts_files) <= 4:
                    image_prompts_files.append(ImagePrompt(cn_img=None))

                hairstyle_lora = None
                selected_hairstyle = params.get('hair_style').lower()   
                if selected_hairstyle == 'quiff':
                    hairstyle_lora = ['quiff.safetensors']
                elif selected_hairstyle == 'pompadour':
                    hairstyle_lora = ['pompadour.safetensors']
                elif selected_hairstyle == 'crew cut':
                    hairstyle_lora = ['crew_cut.safetensors']
                elif selected_hairstyle == 'pixie cut':
                    hairstyle_lora = ['pixie.safetensors']
                elif selected_hairstyle == 'long waves':
                    hairstyle_lora = ['long_waves.safetensors']
                elif selected_hairstyle == 'bob':
                    hairstyle_lora = ['bob.safetensors']
                elif selected_hairstyle == 'combover':
                    hairstyle_lora = ['combover.safetensors']
                elif selected_hairstyle == 'braided':
                    hairstyle_lora = ['braided.safetensors']
                else:
                    hairstyle_lora = []
                
                # Prepare the request object
                req = ImgInpaintOrOutpaintRequestJson(
                    prompt=params.get('prompt', ''),
                    negative_prompt=params.get('negative_prompt', ''),
                    style_selections=params.get('style_selections', []),
                    performance_selection=params.get('performance_selection', 'Speed'),
                    aspect_ratios_selection=params.get('aspect_ratios_selection', '1024*1024'),
                    image_number=params.get('image_number', 1),
                    image_seed=params.get('image_seed', -1),
                    sharpness=params.get('sharpness', 1.0),
                    guidance_scale=params.get('guidance_scale', 7.0),
                    base_model_name='juggernautXL_v8Rundiffusion.safetensors',
                    refiner_model_name=params.get('refiner_model_name', 'None'),
                    refiner_switch=params.get('refiner_switch', 0.8),
                    loras=[Lora(enabled=True, model_name=lora) for lora in hairstyle_lora],
                    advanced_params=advanced_params_obj,
                    save_meta=params.get('save_meta', False),
                    meta_scheme=params.get('meta_scheme', 'fooocus'),
                    save_extension=params.get('save_extension', 'png'),
                    save_name=params.get('save_name', ''),
                    require_base64=params.get('require_base64', True),
                    read_wildcards_in_order=params.get('read_wildcards_in_order', False),
                    async_process=False,
                    uov_input_image=None,
                    uov_method=params.get('uov_method', 'disabled'),
                    upscale_value=params.get('upscale_value', 1.0),
                    image_prompts=[],
                    inpaint_input_image=None,
                    inpaint_additional_prompt=params.get('prompt', ''), ## Note this was changed to be enabled.
                    input_image=input_image_base64,
                    input_mask=input_mask_base64,
                    outpaint_selections=params.get('outpaint_selections', []),
                    outpaint_distance_left=params.get('outpaint_distance_left', 0),
                    outpaint_distance_right=params.get('outpaint_distance_right', 0),
                    outpaint_distance_top=params.get('outpaint_distance_top', 0),
                    outpaint_distance_bottom=params.get('outpaint_distance_bottom', 0)
                )

                # Call the worker to process the image
                from fooocusapi.worker import blocking_get_task_result

                # Process the request
                result = call_worker(req, accept='application/json')

                # Handle the result
                if isinstance(result, list):
                    # Synchronous result
                    image_results = result
                else:
                    # Asynchronous result
                    job_id = result.job_id
                    image_results = blocking_get_task_result(job_id)

                # Assuming single image output
                if image_results and len(image_results) > 0:
                    image_result = image_results[0]
                    image_data_base64 = image_result.base64

                    # Save image to S3
                    result_image_key = f"output_image_{job_id}"
                    s3_client.put_object(
                        Bucket=s3_result_bucket_name,
                        Key=result_image_key,
                        Body=image_data_base64
                    )
                    logger.std_info(f"[SQS Message] Output image saved to S3 with key: {result_image_key}")

                    # Record the time when processing completes
                    completed_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

                    # Retrieve 'created_at' from DynamoDB
                    response = dynamodb_client.get_item(
                        TableName=dynamodb_table_name,
                        Key={'job_id': {'S': job_id}},
                        ProjectionExpression='created_at'
                    )
                    created_at = response['Item']['created_at']['S']

                    # Calculate processing time in seconds
                    processing_time = (
                        datetime.datetime.fromisoformat(completed_time) -
                        datetime.datetime.fromisoformat(created_at)
                    ).total_seconds()

                    # Update DynamoDB with completion details
                    dynamodb_client.update_item(
                        TableName=dynamodb_table_name,
                        Key={
                            'job_id': {'S': job_id}
                        },
                        UpdateExpression='SET job_status = :status, updated_at = :updated_at, completed_at = :completed_at, processing_time = :processing_time',
                        ExpressionAttributeValues={
                            ':status': {'S': 'INPAINT_COMPLETE'},
                            ':updated_at': {'S': completed_time},
                            ':completed_at': {'S': completed_time},
                            ':processing_time': {'N': str(processing_time)}
                        }
                    )

                    logger.std_info(f"[SQS Message] Updated DynamoDB for job_id: {job_id}")

                    # Send message to outbound SQS queue
                    outbound_message = {
                        'job_id': job_id,
                        'result_image_key': result_image_key
                    }
                    sqs_client.send_message(
                        QueueUrl=outbound_queue_url,
                        MessageBody=json.dumps(outbound_message)
                    )
                    logger.std_info(f"[SQS Message] Sent message to outbound queue: {outbound_message}")
                else:
                    logger.std_error("[SQS Message] No image generated.")

                # Delete the processed message from the queue
                sqs_client.delete_message(
                    QueueUrl=inbound_queue_url,
                    ReceiptHandle=receipt_handle
                )

        except Exception as e:
            logger.std_error(f"[SQS Polling] Exception occurred: {e}")
            import traceback
            traceback.print_exc()

            current_time = datetime.datetime.utcnow().isoformat()

            # Update DynamoDB with failure status
            dynamodb_client.update_item(
                TableName=dynamodb_table_name,
                Key={
                    'job_id': {'S': job_id}
                },
                UpdateExpression='SET job_status = :status, updated_at = :updated_at, completed_at = :completed_at',
                ExpressionAttributeValues={
                    ':status': {'S': 'FAILED'},
                    ':updated_at': {'S': current_time},
                    ':completed_at': {'S': current_time}
                }
            )

def sqs_start_polling_thread():
    """
    Start the SQS polling loop in a separate thread.
    """
    sqs_thread = Thread(target=sqs_polling_loop, daemon=True)
    sqs_thread.start()
    logger.std_info("[SQS Polling] SQS polling thread started.")

if __name__ == "__main__":
    logger.std_info(f"[Fooocus API] Python {sys.version}")
    logger.std_info(f"[Fooocus API] Fooocus API version: {version}")
    logger.std_info(f"[Fooocus API] Modified SQS Version 1.0.0")

    parser = argparse.ArgumentParser()
    add_base_args(parser, True)

    args, _ = parser.parse_known_args()
    install_dependents(skip=args.skip_pip)

    from fooocusapi.args import args

    if prepare_environments(args):
        sys.argv = [sys.argv[0]]

        # Load pipeline in new thread
        preload_pipeline_thread = Thread(target=preload_pipeline, daemon=True)
        preload_pipeline_thread.start()

        # Start task schedule thread
        from fooocusapi.worker import task_schedule_loop

        task_schedule_thread = Thread(target=task_schedule_loop, daemon=True)
        task_schedule_thread.start()

        # Start SQS polling thread
        sqs_start_polling_thread()

        # Start api server (if needed)
        from fooocusapi.api import start_app

        start_app(args)

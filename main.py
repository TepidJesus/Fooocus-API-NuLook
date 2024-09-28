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

import boto3  # AWS SDK for Python
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
    from fooocusapi.utils.model_loader import download_models

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

    download_models()

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
    Poll messages from the inbound SQS queue and process them.
    """
    # AWS configuration
    inbound_queue_url = os.environ.get('INBOUND_SQS_URL')
    outbound_queue_url = os.environ.get('OUTBOUND_SQS_URL')
    s3_bucket_name = os.environ.get('S3_BUCKET_NAME')
    s3_result_bucket_name = os.environ.get('S3_RESULT_BUCKET_NAME')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')

    s3_client = boto3.client('s3', region_name=aws_region)
    sqs_client = boto3.client('sqs', region_name=aws_region)

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
                logger.std_info(f"[SQS Message] Received message: {msg}")

                # Extract parameters from the message
                params = msg.copy()
                input_image_key = params.pop('input_image_key', None)
                input_mask_key = params.pop('input_mask_key', None)
                job_id = params.pop('job_id', None)

                # Download and base64-encode input image from S3
                if input_image_key:
                    response = s3_client.get_object(Bucket=s3_bucket_name, Key=input_image_key)
                    input_image_data = response['Body'].read()
                    input_image_base64 = input_image_data
                else:
                    logger.std_error("[SQS Message] No input_image_key provided.")
                    continue

                # Download and base64-encode input mask from S3 (if provided)
                input_mask_base64 = None
                if input_mask_key:
                    response = s3_client.get_object(Bucket=s3_bucket_name, Key=input_mask_key)
                    input_mask_data = response['Body'].read()
                    input_mask_base64 = input_mask_data

                # Set default values for parameters not provided
                from fooocusapi.models.requests_v2 import ImgInpaintOrOutpaintRequestJson, ImagePrompt
                from fooocusapi.models.common.requests import AdvancedParams
                from fooocusapi.routes.generate_v2 import call_worker

                # Create default advanced parameters
                advanced_params = params.get('advanced_params', {})
                advanced_params_obj = AdvancedParams(**advanced_params)

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
                    guidance_scale=params.get('guidance_scale', 7.5),
                    base_model_name=params.get('base_model_name', 'sd_xl_base_1.0'),
                    refiner_model_name=params.get('refiner_model_name', 'None'),
                    refiner_switch=params.get('refiner_switch', 0.8),
                    loras=params.get('loras', []),
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
                    image_prompts=image_prompts_files,
                    inpaint_input_image=None,
                    inpaint_additional_prompt=params.get('inpaint_additional_prompt', ''),
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
                    print(image_data_base64[:100])

                    # Save image to S3
                    image_data = base64.b64decode(image_data_base64)
                    result_image_key = f"output_image_{job_id}.{req.save_extension}"
                    s3_client.put_object(
                        Bucket=s3_result_bucket_name,
                        Key=result_image_key,
                        Body=image_data
                    )
                    logger.std_info(f"[SQS Message] Output image saved to S3 with key: {result_image_key}")

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

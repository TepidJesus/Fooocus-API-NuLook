FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV TZ=Asia/Shanghai

WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python-headless -i https://pypi.org/simple

# Create the directory structure for the model
RUN mkdir -p /app/repositories/Fooocus/models/checkpoints/

# Download the model file
RUN curl -L "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors" \
    -o /app/repositories/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors

EXPOSE 8888

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8888", "--skip-pip", "--preload-pipeline", "--webhook-url", "test" ]

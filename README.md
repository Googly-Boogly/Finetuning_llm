LLM Training Project

This project demonstrates how to train a large language model (LLM) using the autotrain library and various deep learning tools and techniques.
Prerequisites

Before running the training commands, ensure you have the following:

    Python 3.11 or higher
    Docker and Docker Compose

Setup

    Clone this repository and navigate to the project directory.
    Build the Docker image and start the containers:

docker-compose up --build

Install the required Python dependencies:

pip install -r requirements.txt

Update PyTorch:

    autotrain setup --update-torch

Training

To train the LLM, use the following command:

autotrain llm --train --project_name my-llm --model meta-llama/Llama-2-7b-hf --data_path . --use-peft --use_int4 --learning_rate 2e-4 --train_batch_size 6 --num_train_epochs 3 --trainer sft

Explanation of the command arguments:

    --train: Indicates that we want to train the model.
    --project_name: Specifies the name of the project.
    --model: Specifies the pre-trained model to use (in this case, meta-llama/Llama-2-7b-hf).
    --data_path: Specifies the path to the training data.
    --use-peft: Enables the use of Parameter-Efficient Fine-Tuning (PEFT) technique.
    --use_int4: Enables the use of INT4 quantization for reduced memory usage.
    --learning_rate: Sets the learning rate for training.
    --train_batch_size: Specifies the batch size for training.
    --num_train_epochs: Sets the number of training epochs.
    --trainer: Specifies the trainer to use (in this case, sft for Supervised Fine-Tuning).

For more information on available options, run:

autotrain llm --help

Project Structure

    docker-compose.yml: Docker Compose configuration file for setting up the required services.
    Dockerfile: Dockerfile for building the training environment.
    requirements.txt: List of required Python dependencies.
    main.py: Python script for preprocessing the training data.

Note

The training data (data.csv) is not included in this repository. You need to provide your own training data in the specified format.
Acknowledgments

This project utilizes the following libraries and frameworks:

    autotrain
    PyTorch
    Hugging Face Transformers
    PEFT
    bitsandbytes

And this repo:
   
    https://github.com/huggingface/autotrain-advanced
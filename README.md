# Embedding-Autoencoder
Implementation of a Deep Autoencoder, built using Tensorflow v2.x using Tensorflow v1.15 compatibility mode. The model is trained to reduce the dimensionality of the original 1024-sized input data to a 256-sized vector, while simultaneously trying to reconstruct it with the least possible MAE error.

# Run Instructions:
Extract the data from data/routes.rar and run either:
1. Using docker-compose
```sh
docker-compose up (--build)
```
2. Without using Docker: create a new conda/pyenv environment and run:
```sh
pip install -r requirements.txt
cd src
python run.py
```


# Notes:
- Train parameters, such as learning rate and beta1 parameters for the optimizer can be set by the user via the config/config.yaml file.
- In case of execution on a CUDA-enabled machine, to enable GPU acceleration, uncomment line 9 in docker-compose.yml (runtime: nvidia)

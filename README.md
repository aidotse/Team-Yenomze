# Team-Yenmoze Adipocyte Cell Challenge Submission

## CLI tools to reproduce the work

We provide 3 CLI tool to reproduce the docker enviroment, training, testing and PNG outputs of this work. The scripts can be found below and you can run them as in the examples.

### Docker

Docker files are located in `docker` folder in the repository. See the folder tree below:

* **docker**
    * *Dockerfile*
    * *build.sh*
    * *run_docker_notebook.sh*
    * *requirements.txt*
    
You can build the docker image by running the following command in **docker** directory:

```
bash ./build.sh
```

In order to run the image and start jupyter notebook automatically, run following bash script. You need to enter the docker to obtain the jupyter-notebook token to access the notebook after starting the docker. Jupyter-notebook output is directed to `/workspace/docker.log` file in the docker image. you can easily get the token by running `cat /workspace/nohup.out`.

```
bash ./run_docker_notebook.sh
```


### Training

We make a CLI tool to run training as well. The training settings are located in `training_settings` folder and you need to send a `JSON` file to the CLI tool in order to run a training in the given settings.



### Test

In order to load the checkpoints, you need to download `checkpoints` folder on drive, and move the folder to repository root. 

Default checkpoints of the tool is below:

```
BEST_MODELS = {
    "20x": "checkpoints/model20x/G_epoch_548.pth",
    "40x": "checkpoints/model40x/G_epoch_125.pth",
    "60x": "checkpoints/model60x/G_epoch_228.pth"
}
```
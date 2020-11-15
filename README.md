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

Of course, you can start the docker in traditional ways as well :)

```
docker run -it -v /mnt/astra_data:/data -v /path/to/Team-Yenomze-repo:/workspace yenomze:adipocyte
```


### Training

We make a CLI tool to run training as well. The training settings are located in `training_settings` folder and you need to send a `JSON` file to the CLI tool in order to run a training in the given settings. You can run the following command in the **docker environment** to start a training.

```
python train.py -s training_settings/std_train_with_pretrain.json
```

After running this command, the training will start with the standard settings as below:

```
{
    "batch_size": 16, 
    "num_epoch": 500,
    "num_epoch_pretrain_G": 20,
    "lr": 0.00001,
    "unet_split_mode": true,
    "data_dir" :  "/data/*",
    "load_weight_dir": null,
    "save_weight_dir": "checkpoints/standard_training",
    "log_dir": "logs/standard_training",
    "loss_dir": "./lossinfo/standard_training",
    "augmentation_prob": 50,
    "adversarial_weight": 0.05,
    "mse_loss_weight": 50,
    "c01_weight": 0.3,
    "c02_weight": 0.3,
    "c03_weight": 0.4,
    "is_val_split": false    
}
```



### Test

After having the checkpoints for a trained model with our tool, you can load this checkpoint and make&save predictions. You can use our `test.py` CLI tool for that by running following command in **docker environment**. 

```
python test.py -c <checkpoint> -m <magnification_level> -i <input_dir> -o <output_dir>
```

The tool has default paths for our checkpoints. Therefore, if you download our best model checkpoints and move them to our repository root folder, you can run the test simply with following commands:

```
python test.py -m 20x -i /data/20x_images -o outputs/preds/20x_images
```

```
python test.py -m 40x -i /data/40x_images -o outputs/preds/40x_images
```

```
python test.py -m 60x -i /data/60x_images -o outputs/preds/60x_images
```



You need to change the input_dir (-i) to your test image folder that contains only one specific magnification level. You can find our defined best model paths below:

```
BEST_MODELS = {
    "20x": "checkpoints/model20x/G_epoch_548.pth",
    "40x": "checkpoints/model40x/G_epoch_125.pth",
    "60x": "checkpoints/model60x/G_epoch_228.pth"
}
```

### Save Predictions as PNG

You can simply save the TIF (16bits) predictions and ground truths as PNG (8bits) images both in 3 gray channels (C01-02-03) and RGB. The CLI tool that we have for it has very simple approach for it, and it scales the bit values directly to 8 bits. Due to the outlier bit values in the ground data, sometimes they are very darker than out predicted PNG images. We have some cleaning to remove the outliers and learn from more important pixel values. You can find the details for the cleaning approach in Section 2 of our report.

You can run the simple CLI tool with the following command:

```
python save_preds_as_png.py -n <N number of sample> -i <input_dir> -g <grount_truth_dir> -o <output_dir>
```

Example:  
5 random sample from the input directory and save the predictions and ground truth with the following folder structure:

* **output**
    * **png**  
        * **20x**
            * **pred** : this folder contain prediction PNGs
            * **gt** : this folder contain groun truth PNGs

```
python save_preds_as_png.py -n 5 -i output/20x -g /data/20x_images -o output/png
```


## FILE STRUCTURE

* **docker**
    * *Dockerfile*
    * *build.sh*
    * *run_docker_notebook.sh*
    * *requirements.txt*
* **src**
    * **dataloader**
        * *TestDataset.py*
        * *TrainDataset.py*
        * *ValidationDataset.py*
    * **loss**
        * *VGGLoss.py*
    * **model**
        * *Discriminator.py*
        * *Generator.py*
    * **model_handler**
        * *TestHandler.py*
        * *TrainHandler.py*
    * **util**
        * *DataUtil.py*
* **training_settings**
    * *std_train_with_pretrain.json* 
* *train.py*
* *test.py*
* *save_preds_as_png.py*

Place LSUI and UIEB Datasets in `ClearOcean/dataset`

Place Unet backbone checkpoint in `ClearOcean/checkpoints`

Create a CVMIG folder under `ClearOcean/dataset`. You can put images for inference in this folder.

The directory structure after this should be:

```
ClearOcean
├── BasicSR-light
├── checkpoints
│   ├── net_g_latest.pth
├── dataset
│   ├── CVMIG
│   ├── LSUI
│   ├── UIEB
├── experiments
├── PyDiff
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── run.sh
```

Install Docker (https://docs.docker.com/desktop/install/linux/ubuntu/)

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

wget https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb

sudo apt-get update
sudo apt-get install ./docker-desktop-amd64.deb

systemctl --user start docker-desktop
```


Run the following command to run:

```
docker compose build
docker compose up
```

This commands will run the contents of `ClearOcean/run.sh`. Once the training finishes, results will be available at `ClearOcean/experiments`

To change configuration, check the yaml files at `ClearOcean/PyDiff/options`. We have prepared 3 configuration files: 

1. Unet Backbone Training - Train the model with UIEB and LSUI using 128x128 resolution with Unet backbone.

To use this: change the contents of `ClearOcean/run.sh` to

```
python3 pydiff/train.py -opt options/unet_backbone_training.yaml
```


2. Transformer Backbone Training - Train the model with UIEB and LSUI using 128x128 resolution with Transformer backbone

To use this: change the contents of `ClearOcean/run.sh` to

```
python3 pydiff/train.py -opt options/transformer_backbone_training.yaml
```

3. Unet Backbone Inference - Load the best Unet backbone checkpoint and apply inference on the images located at `ClearOcean/dataset/CVMIG`.  For running inference, just add images in this folder and run this configuration. Note that this uses full resolution for the given images.

To use this: change the contents of `ClearOcean/run.sh` to

```
python3 pydiff/train.py -opt options/unet_backbone_inference.yaml
```

Then re-run `docker compose up`.

### repo link: https://github.com/jiaxi312/text-based-recommendation-system/edit/main/README.md

### Required Packages
* tensorflow
* sklearn
* keras-resnet (Note: use the command to install)
`pip install git+https://github.com/vincenzodentamaro/keras-resnet.git`
* Pretrained word embedding matrix GloVe with 42 billion words in 300 dimensions.

### Trained Params
* Resnet34, max sequence length of 750 and output_dim=32. Train MSE loss=0.86, test MSE loss=0.88
https://drive.google.com/drive/folders/12mQApaVDqT98EyMpnznrZ2cBxWIjQHeT?usp=sharing
* Resnet34, max sequence length of 500 and output_dim=64. Train MSE loss=0.8, test MSE loss=0.86
https://drive.google.com/drive/folders/1-9aarSdtghvkBnx0jOWHdAzrH2PVTl6K?usp=sharing
* Resnet34, max sequence length of 500 and output_dim=64. Train MSE loss=0.2, test MSE loss=0.23
with 1 and 0 similarity
* https://drive.google.com/drive/folders/1-5CsYIYV8yQ6x7ffPinWsvBKyenvp5St?usp=sharing


### Training Logs
https://drive.google.com/drive/folders/129RV3nV6N25LATleMFAonix8cWdk2J_4?usp=share_link

### Related Docs
[Project Proposal](https://docs.google.com/document/d/10vavNdC7yXjfwce8LN01kF1wSG4yts0dIxqrc_5muiA/edit#heading=h.w6aqeztfp8bg)

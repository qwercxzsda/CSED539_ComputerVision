# CSED539_ComputerVision

## Train
Download VOC2012 `JPEGImages` folder to `./dataset/voc2012`.

Make folder `./exp/voc2012/pspnet50/model`. This is the path where the trained model will be saved.

Run `python train.py --config config/voc2012/voc2012_pspnet50.yaml` to train the baseline model.

Run `python train.py --config config/voc2012/voc2012_pspnet50_modified.yaml` to train the modified model.

If you want to change settings, please modify `yaml` files located in `./config/voc2012`.

## Test

Train the model first, or download the trained model. Set the path to the trained model `saved_path` in `yaml` files.

Run `python test.py --config config/voc2012/voc2012_pspnet50_test.yaml` to test the baseline model.

Run `python test.py --config config/voc2012/voc2012_pspnet50_modified_test.yaml` to test the modified model.

If you want to change settings, please modify `yaml` files located in `./config/voc2012`.
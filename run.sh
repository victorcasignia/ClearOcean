python3 pydiff/train.py -opt options/unet_no_preprocess.yaml
python3 pydiff/train.py -opt options/unet_histogram_equalization.yaml
python3 pydiff/train.py -opt options/unet_positional_encoding.yaml
python3 pydiff/train.py -opt options/unet_no_channel_attention.yaml
python3 pydiff/train.py -opt options/unet_no_spatial_attention.yaml
python3 pydiff/train.py -opt options/transformer_8_heads.yaml
python3 pydiff/train.py -opt options/transformer_12_heads.yaml
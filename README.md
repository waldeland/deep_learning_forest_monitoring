# deep_learning_forest_monitoring
Estimate vegetation height from satellite images using deep learning.


### Setup and prerequisites
Use a linux computer with CUDA installed and a nvidia-compatible GPU.
 
Install requirments
```
pip install -r requirements.txt
```

### Predict vegetation height
Predict vegetation height for a given Sentinel-2 product:
```
python predict_scene --product S2A_MSIL1C_20170126T073151_N0204_R049_T37MDQ_20170126T074339

```

This will produce a geo-tiff file with 
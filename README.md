# Forest height estimation with deep learning

Code for the paper:<br />
**Forest mapping and monitoring of the African continent using Sentinel-2 data and deep learning**,<br />
Anders U. Waldeland,  Øivind Due Trier,  Arnt-Børre Salberg <br />
[*Preprint submitted for Remote Sensing of Environment*](https://www.journals.elsevier.com/remote-sensing-of-environment)
 


### Setup and prerequisites
Use a linux computer with CUDA installed and a nvidia-compatible GPU.
 
Install requirments
```
pip3 install -r requirements.txt
```

### Usage
Predict vegetation height for a given Sentinel-2 product:
```
python3 predict_scene.py --product S2A_MSIL1C_20170126T073151_N0204_R049_T37MDQ_20170126T074339

```

Run clear cutting detection on a time-series of vegetation height estimates:
```
python3 forest_clear_cutting_detection.py

``` 

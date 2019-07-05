|  dataset  |    aug    | size |   encoder    | decoder |       lr        |  mIoU  |
| :-------: | :-------: | :--: | :----------: | :-----: | :-------------: | :----: |
|   train   | flip(50%) | 512  |  SeResNet50  |  SCSE   |    step0.01     | 0.3146 |
|   train   | flip(50%) | 512  | SeResNext101 |  SCSE   |    step0.01     | 0.2733 |
| train+val | flip(50%) | 512  | SeResNext101 |  SCSE   | warmup+step0.02 | 0.334  |
| train+val | flip(50%) | 512  | SeResNext101 |  SCSE   |    step0.01     | 0.369  |
|   train   | flip(50%) | 512  |   ResNet50   |  SCSE   |    step0.01     | 0.2741 |
|           |           |      |              |         |                 |        |
## TODO

- [ ] mix up
- [ ] CRF
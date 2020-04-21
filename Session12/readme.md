Group Members : Deepak Gowtham, Bikash Ranjan Bhoi


# CIFAR10 With use of Custome Resne and Data Albumentation using One Cycle Policy

## Observation
Best Test Accuracy 90.5%

## LR_FINDER Plot

![LR Finder Plot](https://github.com/bikash-bhoi/eva4/blob/master/Session11/lr_range_test.png)

## ZigZaG Plot

![LR Finder Plot](https://github.com/bikash-bhoi/eva4/blob/master/Session11/zigzag_plot.png)

## Logs

  0%|          | 0/98 [00:00<?, ?it/s]Epoch: 1 Learning_Rate 0.002999999999999999
/content/models/custom_resnet.py:74: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  x=F.log_softmax(x)
Loss=1.3793246746063232 Batch_id=97 Accuracy=36.69: 100%|██████████| 98/98 [00:30<00:00,  3.21it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 1.2689, Accuracy: 5354/10000 (53.54%)

Test acc: 53.54
Epoch: 2 Learning_Rate 0.006955558344737111
Loss=1.3003016710281372 Batch_id=97 Accuracy=52.38: 100%|██████████| 98/98 [00:30<00:00,  3.19it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 1.2353, Accuracy: 5645/10000 (56.45%)

Test acc: 56.45
Epoch: 3 Learning_Rate 0.016504241998412234
Loss=1.0251619815826416 Batch_id=97 Accuracy=56.89: 100%|██████████| 98/98 [00:30<00:00,  3.22it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 1.0076, Accuracy: 6626/10000 (66.26%)

Test acc: 66.26
Epoch: 4 Learning_Rate 0.02605043980435138
Loss=0.8906236290931702 Batch_id=97 Accuracy=62.66: 100%|██████████| 98/98 [00:30<00:00,  3.17it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.9961, Accuracy: 6943/10000 (69.43%)

Test acc: 69.43
Epoch: 5 Learning_Rate 0.029999999881902766
Loss=0.9786694645881653 Batch_id=97 Accuracy=68.62: 100%|██████████| 98/98 [00:30<00:00,  3.19it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.8133, Accuracy: 7420/10000 (74.20%)

Test acc: 74.2
Epoch: 6 Learning_Rate 0.02981559904845503
Loss=0.9494925141334534 Batch_id=97 Accuracy=71.65: 100%|██████████| 98/98 [00:30<00:00,  3.18it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.8510, Accuracy: 7410/10000 (74.10%)

Test acc: 74.1
Epoch: 7 Learning_Rate 0.029268013899015125
Loss=0.8729277849197388 Batch_id=97 Accuracy=74.34: 100%|██████████| 98/98 [00:30<00:00,  3.18it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.6441, Accuracy: 8007/10000 (80.07%)

Test acc: 80.07
Epoch: 8 Learning_Rate 0.02837217987413087
Loss=0.7292106747627258 Batch_id=97 Accuracy=76.21: 100%|██████████| 98/98 [00:30<00:00,  3.20it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.5585, Accuracy: 8142/10000 (81.42%)

Test acc: 81.42
Epoch: 9 Learning_Rate 0.027152530937785864
Loss=0.6896645426750183 Batch_id=97 Accuracy=77.99: 100%|██████████| 98/98 [00:30<00:00,  3.16it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.5547, Accuracy: 8214/10000 (82.14%)

Test acc: 82.14
Epoch: 10 Learning_Rate 0.025642333138551056
Loss=0.5835291147232056 Batch_id=97 Accuracy=79.50: 100%|██████████| 98/98 [00:30<00:00,  3.17it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4844, Accuracy: 8420/10000 (84.20%)

Test acc: 84.2
Epoch: 11 Learning_Rate 0.023882777274732408
Loss=0.49893873929977417 Batch_id=97 Accuracy=79.70: 100%|██████████| 98/98 [00:30<00:00,  3.18it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.5108, Accuracy: 8383/10000 (83.83%)

Test acc: 83.83
Epoch: 12 Learning_Rate 0.021921855411164964
Loss=0.5271540284156799 Batch_id=97 Accuracy=81.92: 100%|██████████| 98/98 [00:31<00:00,  3.15it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4544, Accuracy: 8537/10000 (85.37%)

Test acc: 85.37
Epoch: 13 Learning_Rate 0.01981305189077181
Loss=0.48757070302963257 Batch_id=97 Accuracy=83.16: 100%|██████████| 98/98 [00:30<00:00,  3.18it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4176, Accuracy: 8626/10000 (86.26%)

Test acc: 86.26
Epoch: 14 Learning_Rate 0.01761388454368051
Loss=0.4580290615558624 Batch_id=97 Accuracy=83.26: 100%|██████████| 98/98 [00:30<00:00,  3.20it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4318, Accuracy: 8593/10000 (85.93%)

Test acc: 85.93
Epoch: 15 Learning_Rate 0.01538433588256656
Loss=0.444222629070282 Batch_id=97 Accuracy=84.34: 100%|██████████| 98/98 [00:31<00:00,  3.15it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4384, Accuracy: 8623/10000 (86.23%)

Test acc: 86.23
Epoch: 16 Learning_Rate 0.01318521707353006
Loss=0.39134088158607483 Batch_id=97 Accuracy=85.65: 100%|██████████| 98/98 [00:30<00:00,  3.17it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4382, Accuracy: 8604/10000 (86.04%)

Test acc: 86.04
Epoch: 17 Learning_Rate 0.011076509305366113
Loss=0.4018968343734741 Batch_id=97 Accuracy=85.98: 100%|██████████| 98/98 [00:31<00:00,  3.14it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3829, Accuracy: 8784/10000 (87.84%)

Test acc: 87.84
Epoch: 18 Learning_Rate 0.009115727796550531
Loss=0.4141755700111389 Batch_id=97 Accuracy=86.71: 100%|██████████| 98/98 [00:31<00:00,  3.14it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3437, Accuracy: 8870/10000 (88.70%)

Test acc: 88.7
Epoch: 19 Learning_Rate 0.007356353061816279
Loss=0.3709797263145447 Batch_id=97 Accuracy=87.78: 100%|██████████| 98/98 [00:30<00:00,  3.16it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3333, Accuracy: 8890/10000 (88.90%)

Test acc: 88.9
Epoch: 20 Learning_Rate 0.005846372225684242
Loss=0.41547590494155884 Batch_id=97 Accuracy=87.97: 100%|██████████| 98/98 [00:30<00:00,  3.17it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3327, Accuracy: 8916/10000 (89.16%)

Test acc: 89.16
Epoch: 21 Learning_Rate 0.00462697016876973
Loss=0.30926215648651123 Batch_id=97 Accuracy=88.97: 100%|██████████| 98/98 [00:31<00:00,  3.14it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3231, Accuracy: 8927/10000 (89.27%)

Test acc: 89.27
Epoch: 22 Learning_Rate 0.0037314062059821424
Loss=0.27468809485435486 Batch_id=97 Accuracy=89.71: 100%|██████████| 98/98 [00:31<00:00,  3.15it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3061, Accuracy: 9003/10000 (90.03%)

Test acc: 90.03
Epoch: 23 Learning_Rate 0.0031841069353339816
Loss=0.27790218591690063 Batch_id=97 Accuracy=89.90: 100%|██████████| 98/98 [00:30<00:00,  3.18it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.2991, Accuracy: 9017/10000 (90.17%)

Test acc: 90.17
Epoch: 24 Learning_Rate 0.003
Loss=0.20267312228679657 Batch_id=97 Accuracy=90.21: 100%|██████████| 98/98 [00:31<00:00,  3.16it/s]

Test set: Average loss: 0.2945, Accuracy: 9050/10000 (90.50%)

Test acc: 90.5

Group Members : Deepak Gowtham, Bikash Ranjan Bhoi


# CIFAR10 With use of Regular, Dilated, Depthwise Separable Convolutions

## Observation
Best Test Accuracy 84.05%

### Analysis
Model is still overfiting even after Adding Droupout.

Accuracy is low.

## Logs

 0%|          | 0/782 [00:00<?, ?it/s]Epoch: 1 Learning_Rate [0.020000000000000018]
/content/net.py:97: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Loss=1.4949496984481812 Batch_id=781 Accuracy=38.97: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.96it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 1.3186, Accuracy: 5045/10000 (50.45%)

Epoch: 2 Learning_Rate [0.06583592135001265]
Loss=1.2840418815612793 Batch_id=781 Accuracy=52.48: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.22it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 1.2118, Accuracy: 5802/10000 (58.02%)

Epoch: 3 Learning_Rate [0.18583592135001265]
Loss=0.8772920370101929 Batch_id=781 Accuracy=60.90: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.22it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.9146, Accuracy: 6778/10000 (67.78%)

Epoch: 4 Learning_Rate [0.3341640786499874]
Loss=1.4576996564865112 Batch_id=781 Accuracy=65.56: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.22it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.8598, Accuracy: 7023/10000 (70.23%)

Epoch: 5 Learning_Rate [0.45416407864998737]
Loss=0.933537483215332 Batch_id=781 Accuracy=69.27: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.28it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.7552, Accuracy: 7374/10000 (73.74%)

Epoch: 6 Learning_Rate [0.5]
Loss=0.9702569246292114 Batch_id=781 Accuracy=72.05: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.38it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.7247, Accuracy: 7507/10000 (75.07%)

Epoch: 7 Learning_Rate [0.49373200311754367]
Loss=1.2349026203155518 Batch_id=781 Accuracy=73.98: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.72it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.6635, Accuracy: 7710/10000 (77.10%)

Epoch: 8 Learning_Rate [0.47524231600673683]
Loss=0.8764195442199707 Batch_id=781 Accuracy=75.86: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.52it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.6378, Accuracy: 7807/10000 (78.07%)

Epoch: 9 Learning_Rate [0.445458088785525]
Loss=2.120316743850708 Batch_id=781 Accuracy=77.11: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.20it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.6235, Accuracy: 7879/10000 (78.79%)

Epoch: 10 Learning_Rate [0.40587282697488153]
Loss=0.3936372697353363 Batch_id=781 Accuracy=78.38: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 26.02it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.6143, Accuracy: 7908/10000 (79.08%)

Epoch: 11 Learning_Rate [0.3584715008956504]
Loss=0.646177351474762 Batch_id=781 Accuracy=79.41: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.87it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.5671, Accuracy: 8050/10000 (80.50%)

Epoch: 12 Learning_Rate [0.30563101096814466]
Loss=0.7940012216567993 Batch_id=781 Accuracy=80.42: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.40it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.5595, Accuracy: 8139/10000 (81.39%)

Epoch: 13 Learning_Rate [0.250001]
Loss=0.8898062705993652 Batch_id=781 Accuracy=81.43: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.46it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.5463, Accuracy: 8172/10000 (81.72%)

Epoch: 14 Learning_Rate [0.19437098903185537]
Loss=0.3423202633857727 Batch_id=781 Accuracy=82.70: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.72it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.5249, Accuracy: 8261/10000 (82.61%)

Epoch: 15 Learning_Rate [0.14153049910434962]
Loss=0.3425418436527252 Batch_id=781 Accuracy=83.63: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.87it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.5226, Accuracy: 8230/10000 (82.30%)

Epoch: 16 Learning_Rate [0.09412917302511849]
Loss=0.19848740100860596 Batch_id=781 Accuracy=84.88: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 25.80it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.5054, Accuracy: 8323/10000 (83.23%)

Epoch: 17 Learning_Rate [0.05454391121447502]
Loss=0.2670902907848358 Batch_id=781 Accuracy=86.05: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.25it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.4975, Accuracy: 8353/10000 (83.53%)

Epoch: 18 Learning_Rate [0.024759683993263143]
Loss=0.39819446206092834 Batch_id=781 Accuracy=87.26: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.46it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.4828, Accuracy: 8405/10000 (84.05%)

Epoch: 19 Learning_Rate [0.006269996882456277]
Loss=0.16388359665870667 Batch_id=781 Accuracy=87.94: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:30<00:00, 26.02it/s]
  0%|          | 0/782 [00:00<?, ?it/s]
Test set: Average loss: 0.4808, Accuracy: 8403/10000 (84.03%)

Epoch: 20 Learning_Rate [2e-06]
Loss=0.1741447001695633 Batch_id=781 Accuracy=87.99: 100%|¦¦¦¦¦¦¦¦¦¦| 782/782 [00:29<00:00, 26.53it/s]

Test set: Average loss: 0.4840, Accuracy: 8403/10000 (84.03%)

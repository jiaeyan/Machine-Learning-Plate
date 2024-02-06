### Evaluation  
1. Classification: confusion matrix  

|   | gold positive  |  gold negative |   |   |
|:---:|:---:|:---:|:---:|:---:|
|  system positive |  true positive  |  false positive | $precision=\frac {tp} {tp+fp}$|
|  system negative | false negative  |  true negative |   |   |
|   |  $recall=\frac {tp} {tp+fn}$ |   |  $accuracy=\frac {tp+tn} {tp+fp+tn+fn}$ | $F_1=\frac {2PR} {P+R}$  |

- _P/Precision_: in terms of all the system positives, how many of them are gold positives;  
- _R/Recall_: in terms of all the gold positives, how many of them are system positives;  
- $F_1$: comes from a weighted harmonic mean of precision and recall  

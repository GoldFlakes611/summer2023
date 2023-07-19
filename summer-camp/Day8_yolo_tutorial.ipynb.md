Change `validate.py:L94` to

```
nms_indices = nms(box_corner, probabilities, nms_thres).to("cpu")
```

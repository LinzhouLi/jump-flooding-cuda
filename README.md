# Jump Flooding Algorithm

| ![test](assets/demo.png) | ![test](assets/result.png) | ![test](assets/result_gray.png) |
| ------------------------ | -------------------------- | ------------------------------- |

## Installation

prerequires: pytorch, cuda

install: `python setup.py install`

test: `python demo.py`

## Usage

```python
# data: [H, W], torch.float32, positive pixel value means interior
# result: [H, W, 2], torch.float32, a vector point to nearst pixel, measure in pixel
result = jump_flooding(data)
```


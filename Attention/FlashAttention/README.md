# Flash attention kenrels

We focus on a single batch, single head usecase.

### FA v0
Not flash attention, multiple kernels with excessive HBM memory trasnfers. Using cuBLAS and cuDNN.

Note: the softmax kernel is custom and noticably slow.

### FA V1
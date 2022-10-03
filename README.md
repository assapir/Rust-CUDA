run inside docker:

```bash
docker build -t rust-cuda .
docker run -it --rm --gpus all -v /home/assaf/Code/private/Rust-CUDA:/root/rust-cuda --entrypoint /bin/bash rust-cuda
```

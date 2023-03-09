# NS_KRL

This repo is for our code about the project NS-KRL

## Useage
Use the command:
```
python Main.py --gpu 0 --data WN18RR --name XXX --batch 256 --train_strategy one_to_n --feat_drop 0.2 --hid_drop 0.3 --perm 4 --ker_sz 11 --lr 0.0001 --model_loaded_path None --embed_dim 200 --k_h 20 --k_w 10 --need_n_neg 10
```
or use "CUDA_VISIBLE_DEVICES" like:
```
CUDA_VISIBLE_DEVICES=0 python Main.py --data WN18RR   --name XXX  --lr 0.00003 --batch 256 
```

## MIT License
Copyright (c) 2023 fofilix

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

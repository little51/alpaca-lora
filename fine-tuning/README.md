# 使用deepspeed复现alpaca-lora

## 1、clone代码

```bash
git clone https://github.com/little51/alpaca-lora
cd alpaca-lora
```

## 2、创建虚拟环境

```bash
conda create -n alpaca python=3.10
conda activate alpaca 
```

## 3、安装依赖环境

考虑到github.com的访问问题，从**gitclone.com**镜像安装peft。

### 3.1 安装peft

```shell
pip install git+https://gitclone.com/github.com/huggingface/peft.git -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
 
```

### 3.2 安装其他

```bash
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

### 3.3 升级 accelerate

清华的pip镜像中，accelerate版本是0.19.0，要用以下命令升级到0.20.3以上。

```bash
pip install accelerate -U
```

## 4、训练

```
# 杀进程，用于中途要必参数重新训练
pkill -9 -f  alpaca
# 训练，如果指定具体的GPU，可用--include参数
# 两块P100，用以下参数训练三轮，占用内存15G，需要96个小时
nohup deepspeed --include localhost:1,2 --master_port 12345 finetune.py \
    --base_model ../model/llama_7b \
    --data_path ./fine-tuning/alpaca_gpt4_data_zh.json \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    > alpaca.log 2>&1 &

# 查看日志
tail -f alpaca.log
```

## 5、测试

可以用训练过程中生成的检查点来测试推理效果，如果用8位精度，大约需要9G以上显存。

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --load_8bit \
    --base_model ../model/llama_7b \
    --lora_weights ./lora-alpaca/checkpoint-200
```

如果显存不足，可能会报ValueError: Expected a cuda device, but got: cpu错误，则设置CUDA_VISIBLE_DEVICES=-1，使用CPU测试推理，不过速度非常慢，只能用于测试，不适合用于生产。

## 6、常见问题

### 6.1 基础模型

 如果访问huggingface没有问题，可以用在线模型：--base_model 'decapoda-research/llama-7b-hf' 

如果下载太慢，可以参考https://zhuanlan.zhihu.com/p/633469921的3.2节下载LLaMA原始模型然后转换为huggingface格式，放到本地文件夹中。

### 6.2 bitsandbytes问题

bitsandbytes查找CUDA驱动的机制比较复杂，很多时候查错位置，结果没认出GPU，当成CPU用，会出别的问题，如果在微调时，参考fine-tuning/main.py，修改309、411行直接写成固定的路径，替换/anaconda3/envs/llama-lora/lib/python3.10/site-packages/bitsandbytes/cuda_setup下的main.py。

同时还要关注nocublaslt和非nocublaslt版本的cuda驱动（411行）。


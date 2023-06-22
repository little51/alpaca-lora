# 使用deepspeed训练alpaca-lora

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
CUDA_VISIBLE_DEVICES=1,2 nohup \
deepspeed  --master_port 12345 finetune.py \
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

## 5、常见问题

### 5.1 基础模型

 如果访问huggingface没有问题，可以用在线模型：--base_model 'decapoda-research/llama-7b-hf' 

如果下载太慢，可以参考https://zhuanlan.zhihu.com/p/633469921的3.2节下载LLaMA原始模型然后转换为huggingface格式，放到本地文件夹中。

### 5.2 bitsandbytes问题

bitsandbytes查找CUDA驱动的机制比较复杂，很多时候查错位置，结果没认出GPU，当成CPU用，会出别的问题，如果在微调时，参考fine-tuning/main.py，修改309、411行直接写成固定的路径，替换/anaconda3/envs/llama-lora/lib/python3.10/site-packages/bitsandbytes/cuda_setup下的main.py。

同时还要关注nocublaslt和非nocublaslt版本的cuda驱动（411行）。


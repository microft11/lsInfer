# readme
## 项目简介
项目支持：
- 基于KuiperLLama大模型推理框架的推理
- 基于Qwen2.5/LLama的推理
- 使用cuda，cpu 后端推理

本项目对于KuiperLLama主要更新的点有；
1. ~~添加了 amd 的 rocm 后端实现~~，原作者把cuda和cpu的实现混在一起，太难解耦了。
> rocm后端的实现（WIP）在 https://github.com/microft11/lsInfer/tree/backend_separation 分支中
2. 添加新的模型推理框架支持
3. 添加后端的自动选择
4. 算子的更新优化
5. ~~与某国产企业的推理框架对接~~（但是不商用，且较为机密，估计不会放到这里）

## 第三方依赖
> 借助企业级开发库，更快地搭建出大模型推理框架
1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
3. sentencepiece https://github.com/google/sentencepiece
4. armadillo + openblas https://arma.sourceforge.net/download.html
5. Cuda Toolkit

### 手动下载
```shell
sudo apt-get -y install libgoogle-glog-dev
sudo apt-get -y install libarmadillo-dev
sudo apt-get -y install libsentencepiece-dev # ubuntu版本低了这个库没有需要自己编译
sudo apt-get -y install libgtest-dev
sudo apt -y install libcxxopts-dev
```

## Huggingface镜像站使用
1. 依赖
```bash
pip install huggingface_hub

huggingface-cli login
# cLog in using a token from huggingface.co/settings/tokens
# Create a model or dataset repo from the CLI if needed
huggingface-cli repo create repo_name --type {model, dataset, space}
```

## 模型下载地址
1. LLama2 https://pan.baidu.com/s/1PF5KqvIvNFR8yDIY1HmTYA?pwd=ma8r 或 https://huggingface.co/fushenshen/lession_model/tree/main

2. Tiny LLama 
- TinyLLama模型 https://huggingface.co/karpathy/tinyllamas/tree/main
- TinyLLama分词器 https://huggingface.co/yahma/llama-7b-hf/blob/main/tokenizer.model

3. Qwen2.5/LLama


## 模型导出
```shell
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
# 使用--hf标签从hugging face中加载模型， 指定--version3可以导出量化模型
# 其他使用方法请看export.py中的命令行参数实例
```


## 编译方法
```shell
  mkdir build 
  cd build
  cmake ..
  # 或者开启 USE_CPM 选项，自动下载第三方依赖
  cmake -DUSE_CPM=ON ..
  make -j16
```

## 生成文本的方法
```shell
# ./llama_infer llama2_7b.bin tokenizer.model
./lsInfer/build/demo/llama_infer -m model/llama32_1bnq.bin -t model/tokenizer.model -d cuda
```

# LLama3.2 推理

- 以 meta-llama/Llama-3.2-1B 为例，huggingface 上下载模型：
```shell
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 path/to/tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/llama_infer Llama-3.2-1B.bin meta-llama/Llama-3.2-1B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/llama3_infer.py
```

# Qwen2.5 推理

- 以 Qwen2.5-0.5B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
# pip3 install huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export_qwen2.py Qwen2.5-0.5B.bin --hf=Qwen/Qwen2.5-0.5B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DQWEN2_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/qwen2_infer.py
```

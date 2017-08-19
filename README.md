# 机器阅读理解模型：Deep LSTM Reader

## 简介

复现 [Teaching Machines to Read and Comprehend](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf) 这篇论文中的 Deep LSTM Reader 模型

## 环境要求

代码基于 Python3，安装依赖：

```
$ pip install -r requirements.txt
```

## 运行

保存上下文和词表

```python
python data_utils.py -d /path/to/data/cnn/questions/training -o /your/directory/ -s cnn -v 100000
```

将文本转换成 (context, question, answer) 三元组对象，并保存成 `pickle`

```python
python scripts/convert2index.py -r /path/to/data/cnn/questions/training -d /your/directory/
```

开始训练

```python
python scripts/train.py -r /your/directory/
```
## 说明
模型参考

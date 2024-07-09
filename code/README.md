# README

该目录下，包括我新增的文件和我新修改的文件；LibContinual.zip是加入了我的代码后的整体框架文件  

我已经测试过了，从原来的代码基础上，完成新增和修改后，即可在LibContinual下正常运行我的复现  

## 新增的代码文件

配置文件：inflora.yaml、inflora_b5.yaml、inflora_ca1.yaml  
代码文件：inflora.py、inflora_b5.py、inflora_ca1.py  

可以分别放到 config 和 core/model 目录下  


## 修改的代码文件

因为添加了新方法，所以不可避免的要修改   
**core/model/\_\_init\_\_.py** $\quad$和 $\quad$ **core/model/backbone/\_\_init\_\_.py**  

为了使用我的数据集，还需要修改 **data.yaml** 的数据集路径（data_root）  


因为我的方法涉及ViT模型结构的修改、stage2的训练以及特殊的学习率调度器设置，所以我修改了 **vit.py** 和 **trainer.py**  

我的改动没有加入新功能，都是对原来存在逻辑的补充或者重复。有些改动咨询过助教学长，有些没有，但我个人认为都是合理的，并没有破坏框架的完整性。  

更多细节，详见实验报告  

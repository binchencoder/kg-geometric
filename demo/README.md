# 脚本执行方式

## tkgl_smallpedia_tkg

```bash
# 交互式：逐条输入头实体、关系、时间
python demo/tkgl_smallpedia_tkg.py --mode infer --interactive

# 一次性手动推理（指定时间 2008）
python demo/tkgl_smallpedia_tkg.py --mode infer \
    --head Q648 --relation P27 --time 2008

# 也可用整数 ID
python demo/tkgl_smallpedia_tkg.py --mode infer \
    --head 100 --relation 5 --time 2020 --topk 10

```


## employee_tkg_link_prediction

```bash
# 交互式：逐条输入 头实体 / 关系 / 时间
python examples/employee_tkg_link_prediction.py --mode infer --interactive

# 一次性手动推理（指定时间 2021）
python examples/employee_tkg_link_prediction.py --mode infer \
    --head 张伟 --relation 任职于 --time 2021

# 也可用整数 ID / 关系名
python examples/employee_tkg_link_prediction.py --mode infer \
    --head 0 --relation 0 --time 2022 --topk 10

```
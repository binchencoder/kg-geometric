# 脚本执行方式

## tkgl_example

```bash
# 交互式：逐条输入头实体、关系、时间
python examples/tkgl_example.py --mode infer --interactive

# 一次性手动推理（指定时间 2008）
python examples/tkgl_example.py --mode infer \
    --head Q648 --relation P27 --time 2008

# 也可用整数 ID
python examples/tkgl_example.py --mode infer \
    --head 100 --relation 5 --time 2020 --topk 10

```

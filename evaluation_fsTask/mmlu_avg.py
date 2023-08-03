import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, required=True)
args = parser.parse_args()


with open(args.json_path, 'r') as f:
    data = json.load(f)

# 初始化总 acc 和计数器
acc_sum = 0
count = 0

# 遍历所有 hendrycksTest 相关的数据
for key in data['results']:
    if 'hendrycksTest' in key:
        # 累加 acc 值并增加计数器
        acc_sum += data['results'][key]['acc']
        count += 1

print("Num of tests", count)
# 计算平均值
avg_acc = acc_sum / count

print("mmlu-acc:", avg_acc)
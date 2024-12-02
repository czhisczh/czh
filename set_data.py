import pandas as pd
import numpy as np

# 设定日期范围
dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq='D')

# 设定基础播放量和波动范围
base_play_count = 10000  # 初始播放量
daily_increase_range = (50, 300)  # 每日播放量增加的范围

# 生成播放量数据
play_counts = [base_play_count + np.random.randint(*daily_increase_range) for _ in range(len(dates))]

# 创建 DataFrame
data = pd.DataFrame({
    '日期': dates,
    '播放量': play_counts
})

# 打印前几行数据查看
print(data.head())

# 如果需要保存成 CSV 文件
data.to_csv('youtube_play_counts.csv', index=False)

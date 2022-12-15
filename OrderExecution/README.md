# OrderExecutionVecEnv based on Market Simulator


## File structure

RL training:
- elegantrl  # RL training code for GPU vectorized env
   - run.py  # training loop
   - agent.py  # a collection of PPO algorithms
   - config.py  # configurations (hyper-parameter) for training
   - evaluator.py  # evaluate the cumulative returns of policy network
- demo.py  # understanding the training of single and vectorized env.
- plot.py  # draw the figure for policy evaluation

RL Environment: 
- OrderExecutionEnv.py  # the GPU vectorized env based on Market Simulator
- shares_config.py  # configurations (hyper-parameter) for environment
- shares_classify.py  # use shares with similar turnover and market value to train an agent 
- shares_data_process.py  # process the data of shares to accelerate simulation


## Demo


Run the `demo.py` for training. Read the section "Data" to get the details on data processing.

```
train_ppo_a2c_for_order_execution_vec_env()  # train PPO in OrderExecutionVecEnv

# understanding the training of single and vectorized env.
train_ppo_a2c_for_bipedal_walker()           # train PPO in an OpenAI gym single env
train_ppo_a2c_for_bipedal_walker_vec_env()   # train PPO in an OpenAI gym vectorized env
```


The output of `train_ppo_a2c_for_order_execution_vec_env()`:

```
$ python3 main.py <GPU_ID>                                
| `step`: Number of samples, or total training steps, or running times of `env.step()`.                              
| `time`: Time spent from the start of training to this moment.                                                      
| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode.                            
| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode.                             
| `avgS`: Average of steps in an episode.                                                                            
| `objC`: Objective of Critic network. Or call it loss function of critic network.                                   
| `objA`: Objective of Actor network. It is the average Q value of the critic network.                               
################################################################################                                     
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.                                            
2  2.05e+03     359 |  101.40    0.3  88000     0 |   -2.82   0.39  -0.07  -0.03                                     
2  2.05e+03     359 |  101.40                                                                                        
2  7.27e+04     645 |  102.19    0.3  88000     0 |   -3.08   0.03   0.04   0.12                                     
2  7.27e+04     645 |  102.19                                                                                        
2  7.68e+04     901 |  102.20    0.3  88000     0 |   -3.11   1.39   0.03   0.14                                     
2  7.68e+04     901 |  102.20                                                                                        
2  8.09e+04    1154 |  102.20    0.3  88000     0 |   -3.17   0.01   0.03   0.18                                     
2  1.01e+05    2433 |  102.21    0.3  88000     0 |   -3.40   0.00   0.03   0.29                                     
2  1.10e+05    3008 |  102.23    0.3  88000     0 |   -3.51   0.01   0.03   0.34                                     
2  1.26e+05    4078 |  102.26    0.3  88000     0 |   -3.71   0.13  -2.61   0.45                                     
2  1.42e+05    5132 |  102.27    0.3  88000     0 |   -3.91   0.01   0.05   0.55                                     
2  1.59e+05    6214 |  102.28    0.3  88000     0 |   -4.14   0.01   0.06   0.65                                     
2  1.75e+05    7247 |  102.29    0.3  88000     0 |   -4.28   0.01   0.04   0.73                                     
2  1.87e+05    8037 |  102.27    0.3  88000     0 |   -4.47   0.01   0.04   0.82                                     
2  2.00e+05    8803 |  102.28    0.3  88000     0 |   -4.63   0.57   0.03   0.90                                     
2  2.04e+05    9058 |  102.28    0.3  88000     0 |   -4.68   0.12   0.01   0.93                                     
2  2.20e+05   10088 |  102.28    0.3  88000     0 |   -4.94   0.26  -0.44   1.05      ```
```

### 评估模型的指标 
`avgR` 是平均每股售价 除以 当前平均收盘价 乘以100%：订单执行的的清仓任务完成之后，当天所有卖出的股票的平均出售价格。

这里会使用测试集不同交易日的数据，得到评估结果。测试集的数据是“未来数据”。
因此训练的时候，要保证 Learning curve (y轴是avgR) 保持上升的趋势。
实际训练的时候，把当前能获取到的所有数据放进训练集，并使用 early stop 得到训练完毕的策略网络。

`stdR` 是 `avgR`的标准差。神经网络有时对细微差别敏感，因此环境在reset时产生不同的 目标交易量，为了准确地估计策略网络的表现。
用测试集不同交易日的差异，也是 `stdR` 不等于0的原因之一。 

### 其他可视化评估方法
还可以调用 `plot.py` 里面的函数对训练好的策略进一步评估。详见 章节 `Plot for evaluation`

### early stop 的依据
`expR    objC    objA    advantage_value` 这些数值，都能用来判断训练能否继续。
当这些数值出现预期以外的突变时，训练有可能不能持续。

例如下面展示的 terminal output，`expR` 和 `advantage_value` 在 `7e5` 之后数值突变（之前的数值在小幅度内波动，之后突然偏移）
因此 early stop 的时间点应该在`7e5` 之前。


## Data

#### 数据要求：
- 将数据放在当前工作目录的`shares_data_by_day` 文件夹内
- 按股票代码，把不同股票放在不同的文件夹内 `shares_data_by_day/600000_XSHG`
- 按不同的日期，把相同股票的不同日期的 csv数据放在 `shares_data_by_day/600000_XSHG/2022-01-01.csv`，包含了：
  - DataDate 数据的日期 2022-01-01
  - UpdateTime 数据的时刻 09:30:00
  - LastPrice 统计的时间区间内，最后的成交价
  - Volume 交易量
  - turnover 换手额度（成交量Value）
  - AskPrice1 第1档 买入价格，一般还会记录到5或者10个档位
  - BidPrice1 第1档 卖出价格，一般还会记录到5或者10个档位
  - AskVolume1 挂在交易所的 第1档 买入订单数量
  - BidVolume1 挂在交易所的 第1档 卖出订单数量

#### 数据用法：
- 仿真环境需要这些数据对市场进行模拟。查看 `OrderExecutionVecEnv.load_share_data_dicts()`
- tech factor 也需要根据这些数据生成。查看 `OrderExecutionVecEnv.get_tech_factors()`

#### 数据处理：
所有数据处理的代码，都在 `shares_data_process.py`内。
- `get_share_dicts_by_day()` 获取区间内的交易日后，加载指定区间内的csv数据，并处理成 tensor格式
- `get_trade_dates()` 获取交易日。根据交易日，可以在存放数据的文件夹加载指定日期区间的数据用于训练
- `csv_to_tensor_dict()` 从符合`数据要求`的csv数据中，按天，按股票加载数据，填补缺失值，并处理成 tensor格式
- `fill_zero_and_nan_with_tensor()` 填补缺失值

解释：
- 将数据按不同股票存放，是为了使用风格相近的数据训练同一个智能体
- 将数据按交易日存放，是为了能选取不同时间段的数据用于训练，而不用加载不需要的数据。另外，按天存放，方便按天进行数据归一化处理
- 将数据保存为 pytorch的tensor格式，是为了能够构建 使用GPU有效加速的 Vectorized Env

## Plot for evaluation

### 绘制图片
`check__ask_price_volume()`
- 比较 last price，和策略的 executed price，如果executed price 高，表明策略盈利更多
- 比较 策略挂出的成交数量，和策略的实际成交数量，如果实际成交数量接近挂出的成交数量，且订单分布均匀，证明策略的挂单合理，对市场冲击小

`check__ask_price_volume_with_star()`
- x轴是step数量（大部分step是3秒）
- y轴是价格蓝线last price，并从浅到深的蓝色表示ask和bid价格的档位（1~5档）
- Filled Order 标记为橙色，策略提交的订单数量全部被成交。反之 Not Filled 标记为红色

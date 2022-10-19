# Market Simulator

We simulate the market by 

1). replaying historical limit-order-book-level data;

2). simulating the order matching mechanism.

Currently, the codes work well for minute-level data, i.e., one-step = one minute, which can be altered.

State: a stack of market indicators and market snapshots over the past several time steps.

Action (raw): an order placement. We suppot market orders and limit orders.

Reward: can be configured by the contestant with the aim to generate polices that can optimize pre-specified metrices.

We also provide several wrappers to accept conanoical discret or continuous actions.

We consider the following factors:

1). Temporary market impact;

2). Order delay;

We do NOT consider the following factors yet:

1). permanenet market impact of limit orders;

2). non-resiliency limit-order-book.

## Run

In terminal, 
```bash
    python env.py

```

    


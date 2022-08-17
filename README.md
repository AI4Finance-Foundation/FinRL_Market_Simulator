# Market_Simulator

We simulate the market by replaying historical limit-order-book-level data and simulating the order matching mechanism.

Currently, we simulate at minute-level, i.e., one-step = one minute, which can be altered.

State: a stack of market indicators and market snapshots over the past several time steps.

Action (raw): an order placement. We suppot market orders and limit orders.

Reward: can be configured by the contestant with the aim to generate polices that can optimize pre-specified metrices.

We also provide several wrappers to accept conanoical discret or continuous actions.

We consider the following factors in out simulator:

1). Temporary market impact;

2). Order delay;

We do NOT consider the following factors in our simulator:

1). permanenet market impact of limit orders;

2). non-resiliency limit-order-book.

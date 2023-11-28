Market making is a fundamental trading problem in which
an agent continually quotes two-sided markets for securities, providing
bids and asks that he is willing to trade. Market makers provide liquidity
and depth to markets and earn their quoted spread if both orders are executed.
They are compensated for the risk of holding securities, which may
decline in value between the purchase and the sale. The accumulation of
these unfavourable positions is known as inventory risk. In this paper,
we develop a high-fidelity limit order book simulator based on historical
data with 5 levels on each side, and use it to design a market making
agent using temporal-difference learning. We use a Deep Q-learning as a
value function approximator and a custom reward function that
controls inventory risk.

## Problematic

In this context, the agent observes the current states of the environment
(limit order book, inventory, etc) and uses this information to simultaneously
and repeatedly posts limit orders on both sides at prices derived from an action-space table which affects the next states of the environment and in returns he
receives a reward which is a risk-adjusted function based on its executed orders.
His goal is to learn to act in a way that will maximize its risk-adjusted return
measure over time.
The challenge for the market maker is to mitigate the inventory risk. In particular, this leads the market maker to accumulate  inventory (positive
or negative), before an adverse price move causes him to incur a loss on this inventory.

## Proposed Methodology

In order to design competitive market making agents, leveraging reinforcement learning, for financial markets using high-frequency
historical equities data the steps were 1) a highly realistic order book simulator has been developed, using historical data
from [LOBSTER](https://lobsterdata.com/) combined with our agents’ orders, 2) neural networks were leveraged to represent the policy and value functions allowing
to handle non-linear patterns, tackle low signal-to-noise ratios and manage large
state, which are associated with financial data, 3) agents had the possibility to skew their limit orders to manage inventory risk to help reduce the absolute value of inventory to zero.



## Dataset 
A limit order book (LOB) is a record of outstanding limit orders maintained
by the security specialist who works at the exchange. A limit order is a type of
order to buy or sell a given amount of a security at a specific price or better.

A high-fidelity reconstruction of the limit order book has been developed using high-frequency (up to nanoseconds) historical data. These historical data are provided
by LOBSTER which is an online tool furnishing easy-to-use, high-quality limit
order book data derived from NASDAQ’s Historical TotalView-ITCH. LOBSTER gives access to sample files for multiple tickers (MSFT, AAPL, AMZON,
etc) on 21 June 2021 evolution between 09:30:00 and 15:59:59. These files are
composed of a ’messages’ and an ’orderbook’ file. The ’orderbook’ file contains
the evolution of the LOB up to the requested number of level (5 in this work).
The ’message’ file contains indicators for the type of event (order) causing an
update of the LOB in the requested price range. The type of event can be: 1)
Submission of a new limit order, 2) Cancellation (partial deletion of a limit order), 3) Deletion (total deletion of a limit order). Each event is characterized by
an id, a size (volume), the price and the direction (buy or sell).

## Trading Strategy
The market making agent acts every second on events as they occur on the LOB.
An event is anything that constitutes an observable change in the states of the
environment (change in mid price, spread, best buy price, etc). The agent is
restricted to a single buy and sell order, per second, with an order size of 100,
and cannot exit the market. Given that the market is built based on past data,
any orders simulated by the agent would not have any impact on the market.
Hence, small order size of 100 was chosen, which in comparison to the overall
trading volume of the market has a negligible impact.

Actions 0 to 8 are limit orders to be placed at fixed distances relative to the mid-price, with Action  ∈ (0, 4)  leading to buy-sell quotes that are increasingly further to the top of the book and Action ∈ (5, 8) allow to skew the limit orders in favour of a side.
Action 9 is a market order designs to bring the agent closer to a neutral position if the inventory constraints are no longer satisfied.

## Reward Function
An idealised market maker seeks to make money by repeatedly making its quoted spread, while keeping its inventory low to minimise market exposure.
The “natural” reward function for trading agents is the PnL which represents how much
money is made or lost through exchanges with the market.
This basic formulation of PnL, captures both the gain from speculation and
the gain from spread, which can lead to instability during learning due to
inventory risk.
Another reward function called 'Asymmetrically Dampened PnL' has been engineered to encourage market making behaviour by reducing
the reward that the agent can gain from speculation, thanks to the dampening
factor η which reduces only the profits from speculation.

## State Representation
The environment is composed of two states, the agent-state and the market-state. These states are constructed as set of attributes that describe the agent
and the market at time t. The agent state is composed of the inventory and the quoting distances (trading strategy) while the market state is composed of the spread, book imbalance, volatility, relative strength index and other handcrafted features.
These technical indicators derived from the orderbook attributes and
added to the market state, represent statistical tools and are extensively
use to make investment decisions by generating signals. They help to identify
trends, regime switches, momentum, and potential reversal points.

## Policy

Market prices are Markovian in nature, i.e. the probability distribution of the
price of an asset depends only on the current price and not on the prior history.
Moreover, states in the context of market making are a great source of complexity, the market state is in constant evolution, a multi-agent system and data
can be unrepresentative (black swan), hence, the state the agent observes is some derivation of the true state.
It makes sense to formulate this problem as a Partially Observable Markov
Decision Process (POMDP) problem. 
The goal of reinforcement learning algorithms is to find an optimal policy π : S → A, i.e. a function mapping between
states and actions such that it maximizes the expected long-term reward, known
as the Q-value.
Neural networks with their universal approximation capabilities are a natural
choice to find a function Qθ(s, a) that approximates Qπ(s, a), the inputs are the
states of the environment and the outputs are the Q-values for each state-action pair.
The optimal policy is defined trivially as follows: the agent will select the action with the largest predicted Q-Value.
The target Q-Value should be close to the reward the agent gets after playing
action a in state s plus the future discounted value of playing optimally from
then on.
The loss to be minimized is the mean squared error between the estimated Q-Value and the target Q-Value.

## DQN 

The networks are composed of Linear() or LSTM() layers. Linear() layers
are followed by ReLU() activation. In the case of LSTM() layers, the activation
function used is Tanh() and the state covers the last 60 time-steps (one time-step
= one second) to represent the sequential nature of the problem. For both types,
a dropout() layer with probability = 0.1 is added before the hidden layer. The
numbers of neurons per layer is 256 in order to capture the complexity of the
problem. Furthermore, the Adam optimizer, with its default parameters, is used
for training with a learning rate of 0.001

## Results

The policy these agents were able to learn led to competitive out-of-sample performance and demonstrates superior performance over several standard Market
Making benchmark strategies. The application of reinforcement learning to such
a problem also shows that it is capable of working well in complex environments
and that it is a viable tool that can be used for market making.

For more details, please refer to the paper. 

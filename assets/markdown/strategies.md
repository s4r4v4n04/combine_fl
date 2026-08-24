# Federated Learning Strategies

Flotilla's **Strategies** are highly modular, combining **Aggregation** (global model updates) with **Client Selection** (participant filtering).

Explore the built-in strategies below, or easily plug in your own custom modules.

> **Note:** All aggregation strategies below are implemented for every supported ML framework (PyTorch, TensorFlow, JAX, Scikit-Learn, ONNX). Select the appropriate variant by appending the framework suffix to the aggregator name in your training config (e.g., `fedavg_torch`, `fedavg_tensorflow`, `fedavg_jax`, `fedavg_sklearn`, `fedavg_onnx`).

<br/>

## FedAvg <picture><img src="https://img.shields.io/badge/Sync-374151?style=flat-square" alt="Sync" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Aggregation-2563EB?style=flat-square" alt="Aggregation"></picture> [`aggregator_fedavg_torch.py`](../../src/server/aggregation/aggregator_fedavg_torch.py)<br/>
  *Synchronously aggregates global models by computing a weighted average of client updates scaled by their local dataset sizes.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_fedavg.py`](../../src/server/clientselection/client_selection_fedavg.py)<br/>
  *Selects a random subset of active clients for standard Federated Averaging.*
* <picture><img src="https://img.shields.io/badge/Multi--Framework-10B981?style=flat-square" alt="Multi-Framework"></picture> FedAvg aggregation is available for all supported frameworks.

> **Reference:** *Communication-Efficient Learning of Deep Networks from Decentralized Data* — H. Brendan McMahan, et al. (AISTATS 2017). <br/>
> <a href="https://arxiv.org/abs/1602.05629"><picture><img src="https://img.shields.io/badge/arXiv-1602.05629-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></picture></a>

<hr>

## FedAsync <picture><img src="https://img.shields.io/badge/Async-374151?style=flat-square" alt="Async" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Aggregation-2563EB?style=flat-square" alt="Aggregation"></picture> [`aggregator_fedasync_torch.py`](../../src/server/aggregation/aggregator_fedasync_torch.py)<br/>
  *Asynchronously updates the global model upon receiving a client report, applying a polynomial staleness penalty based on model version delay.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_fedasync.py`](../../src/server/clientselection/client_selection_fedasync.py)<br/>
  *Asynchronously selects available returning clients individually, optimizing for high-throughput stream processing.*
* <picture><img src="https://img.shields.io/badge/Multi--Framework-10B981?style=flat-square" alt="Multi-Framework"></picture> FedAsync aggregation is available for all supported frameworks.

> **Reference:** *Asynchronous Federated Optimization* — Cong Xie, Sanmi Koyejo, Indranil Gupta. (OPT 2020). <br/>
> <a href="https://arxiv.org/abs/1903.03934"><picture><img src="https://img.shields.io/badge/arXiv-1903.03934-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></picture></a>

<hr>

## FedAT <picture></picture> <picture><img src="https://img.shields.io/badge/Async-374151?style=flat-square" alt="Async" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Aggregation-2563EB?style=flat-square" alt="Aggregation"></picture> [`aggregator_fedat_torch.py`](../../src/server/aggregation/aggregator_fedat_torch.py)<br/>
  *A tier-based strategy that performs synchronous FedAvg within speed-based tiers, and asynchronous weighted aggregation across different tiers.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_fedat.py`](../../src/server/clientselection/client_selection_fedat.py)<br/>
  *Groups clients into performance tiers to orchestrate synchronous intra-tier and asynchronous inter-tier federated learning loops.*
* <picture><img src="https://img.shields.io/badge/PyTorch_Only-EF4444?style=flat-square" alt="PyTorch Only"></picture> FedAT aggregation is currently implemented for PyTorch only.

> **Reference:** *FedAT: A High-Performance and Communication-Efficient Federated Learning System with Asynchronous Tiers* — Zheng Chai, et al. (SC 2021). <br/>
> <a href="https://arxiv.org/abs/2010.05958"><picture><img src="https://img.shields.io/badge/arXiv-2010.05958-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></picture></a>

<hr>

## TiFL & TiFL Lite <picture><img src="https://img.shields.io/badge/Sync-374151?style=flat-square" alt="Sync" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Aggregation-2563EB?style=flat-square" alt="Aggregation"></picture> [`aggregator_fedavg_torch.py`](../../src/server/aggregation/aggregator_fedavg_torch.py)<br/>
  *Synchronously aggregates global models by computing a weighted average of client updates scaled by their local dataset sizes.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_tifl.py`](../../src/server/clientselection/client_selection_tifl.py)<br/>
  *Groups clients into tiers by training latency and adjusts selection probabilities dynamically based on tier-wise test accuracy to mitigate stragglers.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_tifl_lite.py`](../../src/server/clientselection/client_selection_tifl_lite.py)<br/>
  *A simplified, lower-overhead version of the tier-based TiFL selection algorithm.*

> **Reference:** *TiFL: A Tier-based Federated Learning System* — Zheng Chai, et al. (HPDC 2020). <br/>
> <a href="https://arxiv.org/abs/2001.09249"><picture><img src="https://img.shields.io/badge/arXiv-2007.05171-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></picture></a>

<hr>

## HACCS <picture><img src="https://img.shields.io/badge/Sync-374151?style=flat-square" alt="Sync" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Aggregation-2563EB?style=flat-square" alt="Aggregation"></picture> [`aggregator_fedavg_torch.py`](../../src/server/aggregation/aggregator_fedavg_torch.py)<br/>
  *Synchronously aggregates global models by computing a weighted average of client updates scaled by their local dataset sizes.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_haccs.py`](../../src/server/clientselection/client_selection_haccs.py)<br/>
  *Uses hierarchical agglomerative clustering to select clients by intelligently balancing local dataset distributions and training latency tradeoffs.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_haccs_lite.py`](../../src/server/clientselection/client_selection_haccs_lite.py)<br/>
  *A lighter variant of the HACCS algorithm that performs fast agglomerative clustering over data distributions.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_haccs_lite_deprecated.py`](../../src/server/clientselection/client_selection_haccs_lite_deprecated.py)<br/>
  *An older, deprecated implementation of the HACCS Lite client selection algorithm.*

> **Reference:** *[HACCS Paper Title]* — Joel Wolfrath, et al. (IPDPS 2022). <br/>
> <a href="https://doi.org/10.1109/IPDPS53621.2022.00100"><picture><img src="https://img.shields.io/badge/DOI-10.1109%2FIPDPS53621.2022.00100-0077b6?style=flat-square" alt="DOI"></picture></a>

<hr>

## Loss-Based Strategies <picture><img src="https://img.shields.io/badge/Sync-374151?style=flat-square" alt="Sync" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_high_loss.py`](../../src/server/clientselection/client_selection_high_loss.py)<br/>
  *Selects a specified fraction of clients strictly based on the highest reported validation or training loss to prioritize learning from harder examples.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_probabilistic_high_loss.py`](../../src/server/clientselection/client_selection_probabilistic_high_loss.py)<br/>
  *Probabilistically selects clients, assigning those with higher reported losses a proportionally higher chance of being chosen for the round.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_low_loss.py`](../../src/server/clientselection/client_selection_low_loss.py)<br/>
  *Selects clients strictly based on the lowest reported loss, focusing on stable updates or confident local models.*

<hr>

## Unbiased & Heuristic Strategies <picture><img src="https://img.shields.io/badge/Sync-374151?style=flat-square" alt="Sync" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_random_subset.py`](../../src/server/clientselection/client_selection_random_subset.py)<br/>
  *Uniformly selects a random fraction of the available active clients without relying on any specific local metrics.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_reliable_fedavg.py`](../../src/server/clientselection/client_selection_reliable_fedavg.py)<br/>
  *Extends basic random selection by filtering and prioritizing clients that exhibit highly reliable network or resource stability profiles.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_all_clients.py`](../../src/server/clientselection/client_selection_all_clients.py)<br/>
  *Selects all active and available clients unconditionally for the training round.*

<hr>

## Scheduling Strategies <picture><img src="https://img.shields.io/badge/Sync-374151?style=flat-square" alt="Sync" style="vertical-align: middle;"></picture>

* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_odd_even.py`](../../src/server/clientselection/client_selection_odd_even.py)<br/>
  *Alternates client selection based on odd or even round numbers for a simple, predictable cyclic partitioning of the client pool.*
* <picture><img src="https://img.shields.io/badge/Client_Selection-0891B2?style=flat-square" alt="Client Selection"></picture> [`client_selection_tiered_roundrobin.py`](../../src/server/clientselection/client_selection_tiered_roundrobin.py)<br/>
  *Clusters clients into performance tiers and applies a round-robin schedule to iteratively select active subsets across these tiers.*

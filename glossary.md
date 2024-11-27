# Table of contents
1. [Contract types](#contract-types)
   - [Preliminary concepts](#preliminary-concepts)
     - [Contracted energy](#contracted-energy)
     - [Hybrid generation profiles](#hybrid-generation-profiles)
   - [Pay as Consumed](#pay-as-consumed)
   - [Pay as Produced](#pay-as-produced)
   - [24/7](#247)
   - [Baseload](#baseload)
   - [Shaped](#shaped)
2. [Data labels](#data-labels)
   - [Dataframe columns](#dataframe-columns)
     - [Contracted Energy](#contracted-energy-column)
     - [Hybrid](#hybrid-column)
     - [Load](#load-column)
     - [RRP](#rrp-column)
     - [AEI](#aei-column)

---

# Contract types

The five contract types the PPA Analysis tool functionality can handle are described here. However, two key concepts for understanding all contracts are outlined first. 

## Preliminary concepts

### Contracted energy
The contract type determines how the **contracted energy**, the amount of energy to be traded through a power purchase agreement (PPA), is set. Within the PPA analysis tool framework, the contracted energy is recorded in the "Contracted Energy" column of the time series dataframes.

### Hybrid generation profiles
The other important quantity for understanding the contract types is the **hybrid generation profile**. A hybrid generation profile is the sum of all the generation produced by contracted generators, multiplied by their respective contracted percentages, on an interval-by-interval basis. For example, if two generators (A and B) are contracted with 15% and 40% of their capacity under contract respectively, and in a given interval A generates 40 MWh and B generates 70 MWh, the hybrid energy for that interval would equal:
- \( 0.15 \times 40 + 0.40 \times 70 = 34 \, \text{MWh} \)

The term **hybrid** is used to describe this quantity because it represents a generation profile that is a hybrid of multiple generators' individual profiles. Within the PPA analysis tool, this hybrid quantity is recorded in the "Hybrid" column of the time series dataframes.

## Contracts

### Pay as Consumed
For the **Pay as Consumed** contract type, the contracted energy is a time-varying quantity which is the lesser of the [hybrid](#hybrid-generation-profiles) volume and [load](#load) consumption volume on an interval-by-interval basis. The contracted energy volume determines how the following PPA costs are calculated: 

- **Contract for difference settlement costs:** These are calculated as the contracted energy multiplied by the difference between the strike price and the wholesale spot price on an interval-by-interval basis.  
  Formula: `settlement cost = contracted energy * (strike price - wholesale spot price)`
  
- **Firming costs:** Any load volume not covered by the contracted energy volume is purchased at the [Firming price](#firming-price) on an interval-by-interval basis.

- **Production shortfall revenue:** If the total contracted energy over the settlement period of the contract does not equal or exceed a guaranteed percentage of the load, a shortfall penalty applies. The revenue from the penalty is equal to the shortfall volume multiplied by the penalty rate.  
  Formula: `shortfall revenue = ((total load * guaranteed percentage) - contracted energy) * penalty rate`
  - Note: The penalty only applies when contracted energy < (total load * guaranteed percentage).

- **Cost from [LGC](#lgc) undersupply:** The volume of LGCs delivered through the PPA contract is determined by the total contracted energy across the settlement period. If the contracted energy is less than the guaranteed percentage of the load, load is assumed to source LGCs from a third party at the LGC buy price.  
  Formula: `LGC cost = ((total load * guaranteed percentage) - contracted energy) * LGC buy price`
  - Note: This only applies when contracted energy < (total load * guaranteed percentage).

### Pay as Produced
For the **Pay as Produced** contract type, the contracted energy is a time-varying quantity which is always equal to the [hybrid](#hybrid-generation-profiles) volume. The **contract for difference settlement**, **firming**, **production shortfall**, and **LGC undersupply** costs are determined as for the [Pay as Consumed](#pay-as-consumed) contract type. Additionally, because for the Pay as Produced contract the contracted energy can be greater than the load volume, two other contract costs can apply:

- **Revenue from [LGCs](#lgc) oversupply:** If the contracted energy exceeds the guaranteed percentage of the load, LGCs delivered in excess of this guaranteed load are assumed to be available for on-sale at the LGC sell price.  
  Formula: `excess LGC revenue = (contracted energy - (total load * guaranteed_percentage)) * LGC sell price`
  - Note: This only applies when contracted energy > (total load * guaranteed percentage).

- **Excess energy revenue:** Any contracted energy in excess of the [load](#load), on an interval-by-interval basis, can be on-sold by the buyer, either at the wholesale spot price or at a specified fixed price.

### 24/7
The **24/7** contract is very similar to the Pay as Consumed contract, with the contracted energy volume in each interval being the minimum of the load and hybrid volumes (i.e., all generated energy up to load is contracted). The key difference is how the PPA analysis tool handles the optimization of contracting. For the Pay as Consumed contract, the fraction of each generator's contracted capacity is optimized to minimize the cost of purchasing energy, assuming energy is bought from generators at their LCOE and that load not covered by the generators is purchased at the wholesale spot price. For the 24/7 contract, an additional penalty is applied in the optimization to incentivize close temporal matching between generation and load. 

A matching percentage (the percentage of load covered by generation, capped at 100%) is calculated for each interval. If the average matching percentage falls below the percentage of load specified to be covered in the contract, a penalty applies per MWh of shortfall. Additionally, the **production shortfall revenue** in this contract is calculated similarly, but the user has the option to supply different target matching percentages for the optimization and guaranteed percentage for the shortfall revenue.

### Baseload
The **Baseload** contract uses a constant contracted energy volume on a yearly, quarterly, or monthly basis, which is set equal to the average load for the period, multiplied by the contract volume percentage. The cost components for the **contract for difference settlement**, **firming**, **LGC undersupply**, **LGC oversupply**, and **excess energy** are calculated as for the Pay as Consumed and Pay as Produced contracts. However, the **production shortfall revenue** is calculated on an interval-by-interval basis, with the penalty payable whenever the hybrid volume is less than the contracted energy volume.

### Shaped
The **Shaped** contract is similar to the Pay as Produced contract, but rather than the contracted energy volume equaling the hybrid volume, it is set to a synthetic generation profile calculated on a yearly, quarterly, or monthly basis. This profile aims to replicate the patterns (or "shapes") of generation. To calculate the synthetic generation profile, the user specifies a percentile, and for each hour in the day, the percentile of generation in that hour across the period (year, quarter, or month) is calculated. For example, the 0.5 percentile is the median generation for that hour of the day across the period. A daily profile is constructed by combining the percentiles of generation for each hour of the day.

---

# Data labels

## Dataframe columns
Key dataframe columns used throughout the tool are described here.

### Contracted Energy column
The volume of load to be covered by the PPA in each interval. It can also be thought of as the volume of renewable energy being purchased via the PPA in each interval.

### Hybrid column
Contains the volume of energy generated by the optimized hybrid renewable energy portfolio in each interval. This column is calculated as the weighted sum of each generator's output in the portfolio by its optimized percentage.

### Load column
This is the volume of energy consumed by the buyer in each interval.

### RRP column
Contains the wholesale energy prices for each interval in the same energy market region as the load. Units are $/MW/h.

### AEI column
This column holds the Average Emissions Intensity (AEI) factors for each interval in the same energy market region as the load. Units are tCO2-e/MWh.

# Table of contents

# Contract types

The five contract types the PPA Analysis tool functionality can handle are described here. However, firstly two key 
concepts for understanding all contracts are outlined. 

## Preliminary concepts

### Contracted energy

The contract type determines how the contracted energy, the amount of energy to be traded through power purchase 
agreement is set. With in the PPA analysis tool framework the contracted energy is recorded in the "Contracted Energy"
column of the time series dataframes.

### Hybrid generation profiles

The other import quantity for understanding the contract types is the hybrid generation profiles, a hybrid 
generation profile is the sum of  all the generation produced by contracted generators multiplied by their respective 
contracted percentages, on an interval by interval basis. For example, if two generators A and B where contracted, with 
15 % and 40 % of their capacity under contract, respectively, and in a given interval A generated 40 MWh and B 
generated 70 MWh, then the hybrid energy for the interval would equal 0.15 * 40 + 0.40 * 70 = 34 MWh. The term hybrid is 
used to describe this quantity because it is a generation profile that is hybrid between of multiple generators 
individual profiles. With in the PPA analysis tool framework the hybrid quantity is recorded in the "Hybrid" column of 
the time series dataframes.

## Contracts:

### Pay as Consumed

For the Pay as Consumed contract type the contracted energy is a time varying quantity which is the lesser of the
[hybrid](#Hybrid-generation-profiles) volume and [load](#Load) consumption volume on an interval by interval basis. 
The contracted energy volume determines how the following PPA costs are determined: 

***Contract for difference settlement costs:*** are calculated as the contracted energy multiplied by the difference 
between the strike price and the wholesale spot price on an interval by interval basis i.e. settlement cost = 
contracted energy * (strike price -  wholesale spot price)

***Firming costs:*** Any load volume not covered by the contracted energy volume, on an interval by interval basis, is 
purchased at the [Firming price](#firming-price).

***Production shortfall revenue:*** If the total contracted energy over the settlement period of the contract does
not equal of exceed a guaranteed percentage of the load then a shortfall penalty applies, with the revenue from the 
penalty equal to the short fall volume multiplied by the penalty rate i.e. 
shortfall revenue = ((total load *  guaranteed percentage) - contracted energy) * penalty rate. Note, the penalty 
only when contracted energy < (total load *  guaranteed percentage).

***Cost from [LGC](#LGC) undersupply*** The volume of LGCs delivered through the PPA contract is determined by the total
contracted energy across the settlement period. If the contracted energy is less than the guaranteed percentage of the 
load, then load is assumed to source LGCs from a third party at the LGC buy price, creating an LGC cost = ((total 
load * guaranteed_percent) - contracted energy) * LGC buy price. Note, this only applies when 
contracted energy < (total load *  guaranteed percentage).

## Pay as Produced

For the Pay as Produced contract type the contracted energy is a time varying quantity which is always equal the
[hybrid](#Hybrid-generation-profiles) volume. The ***contract for difference settlement***, ***firming***, 
***production shortfall***, and ***LGC undersupply*** costs are determined as for [Pay as Consumed](#pay-as-consumed)
contract type. Additionally, because for the Pay as Produced contract the contracted energy can be greater than the load
volume two other contract costs can apply:

***Revenue from [LGCs](#LGC) oversupply*** The volume of LGCs delivered through the PPA contract is determined by the total
contracted energy across the settlement period. If the contracted energy exceeds the guaranteed percentage of the load, 
then LGCs delivered in excess of this guaranteed load are assumed to be available for on sale at the LGC sell price, 
creating excess LGC revenue = (contracted energy - (total load * guaranteed_percent)) * LGC sell price. Note, this 
only applies when contracted energy > (total load *  guaranteed percentage).

***Excess energy revenue:*** It is assumed any contracted energy in excess of the [load](#load), on an interval by 
interval basis, can be on sold by the buyer, either at the wholesale spot price or at a specified fixed price.

## 24/7

The 24/7 contract is very similar to the Pay as Produced contract, with the contracted energy volume always equal to the
hybrid volume. The key difference between the two contract types is how the PPA analysis tool handles the optimisation
of the contracting. For the Pay as Produced contract the fraction of each generator's contracted capacity is 
optimised to minimise cost of purchasing energy, assuming energy is bought from generator's at their LCOE, and that 
load not covered by the generators is purchased at the wholesale spot price, and the total volume from renewables is 
constrained to be equal to or greater than a specified fraction of the total load over the optimisation period. However,
for the 24/7 contract, an additional penalty is also applied in the optimisation to incentivise close temporal
matching between generation and load. A matching percentage, the percentage of load covered by generation (capped at 
100%) is calculated for each interval, and if the average matching percentage falls below the percentage of the load 
specified to be covered in the contract then a penalty applies per MWh of shortfall. Additionally, the ***production 
shortfall revenue*** in contract is calculated in the same way, but the user has the option to supply different target
matching percentages (contracted volume for the optimisation, and guaranteed percentage for the shortfall revenue), 
and the penalty rate for the optimisation is set by a default value in the advanced_settings.module, while the 
shortfall penalty is provided directly to the calculate_bill function.

## Baseload

The Baseload contract uses a constant contracted energy volume, on yearly, quarterly, or monthly basis, which is set 
equal to the average load for the period, multiplied by the contract volume percentage. The cost components for the 
***contract for difference settlement***, ***firming***, ***LGC undersupply***, ***LGC 
oversupply***, and ***excess energy*** are calculated as for Pay as Consumed and Pay as Produced contracts. While the
***production shortfall revenue*** is calculated on an interval by interval basis with the penalty payable whenever 
the hybrid volume is less than the contracted energy volume.

## Shaped

The Shaped contract is similar to the Pay as Produced contract, but rather than the contracted energy volume equalling
the hybrid volume it is set to synthetic generation profile calculated on yearly, quarterly, or monthly basis, which 
aims to replicate the patterns (or shapes) of generation. To calculate the synthetic generation profile, the users 
specifies a percentile, and then for each hour in the day the percentile of generation in that hour across the period 
(year, quarter, or month) is calculated, i.e. 0.5 percentile is the median generation in that hour of the day across 
the period. A daily profile for each period is constructed by combining the percentiles of generation for each hour 
of the day.

# Data labels

## Dataframe columns

Key dataframe columns used throughout the tool are described here.

### Contracted Energy

The volume of load to be covered by the PPA in each interval. Can also be thought of as the volume of renewable energy 
being purchased via the PPA in each interval. 

### Hybrid

### Load

### RRP

### AEI

### <DUID>: <Technology>

### Firming price

# Miscellaneous technical terms

## LGC

## Guaranteed percentage




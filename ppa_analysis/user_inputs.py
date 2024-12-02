import copy
import functools
import logging
import os
import textwrap

import ipywidgets as widgets
import pandas as pd
from IPython.display import HTML, display
from nemosis import static_table

from ppa_analysis import (
    advanced_settings,
    firming_contracts,
    helper_functions,
    import_data,
    tariffs,
)

logging.getLogger("nemosis").setLevel(logging.WARNING)


def on_toggle_click(which_widg, text, change):
    # with out:
    if change["new"]:
        which_widg.value = "<br>".join(textwrap.wrap(text, 50))
    else:
        which_widg.value = ""


def button_setup(widget_input, help_text, label_text):
    help_text_label = widgets.HTML(value="")
    help_button = widgets.ToggleButton(
        description="(?)",
        disabled=False,
        button_style="",
        icon="",
        layout=widgets.Layout(width="40px", height="28px"),
    )
    help_button.style.button_color = "grey"
    help_button.observe(
        functools.partial(on_toggle_click, help_text_label, help_text), "value"
    )

    widget = widgets.VBox([widget_input])
    widget_btn = widgets.VBox([help_button])
    widget_lbl = widgets.VBox(
        [
            widgets.Label(
                label_text,
                layout=widgets.Layout(display="flex", justify_content="flex-end"),
            )
        ]
    )
    widget_help = widgets.VBox([help_text_label])

    hbox_collection = widgets.HBox([widget_lbl, widget, widget_btn, widget_help])

    return display(hbox_collection)


def launch_input_collector():
    display(
        HTML(
            """
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 40ex !important; }
        .widget-dropdown select { min-width: 40ex !important; }
        .widget-floattext input { min-width: 40ex !important; }
        </style>
        """
        )
    )

    display(
        HTML(
            """
        <h3>Historical data selection:</h3>
        """
        )
    )

    out = widgets.Output()

    input_collector = {}

    years_with_cached_data = helper_functions.get_data_years(
        advanced_settings.YEARLY_DATA_CACHE
    )
    input_collector["year"] = widgets.Dropdown(
        options=years_with_cached_data,
        value=years_with_cached_data[0],
        description="Year:",
        disabled=False,
    )
    display(input_collector["year"])

    input_collector["generator_region"] = widgets.Dropdown(
        options=advanced_settings.NEM_REGIONS,
        value=advanced_settings.NEM_REGIONS[0],
        description="Generator region:",
        disabled=False,
    )
    display(input_collector["generator_region"])

    input_collector["load_region"] = widgets.Dropdown(
        options=advanced_settings.NEM_REGIONS,
        value=advanced_settings.NEM_REGIONS[0],
        description="Load region:",
        disabled=False,
    )
    display(input_collector["load_region"])

    input_collector["load_data_file"] = widgets.Dropdown(
        options=[
            os.path.splitext(fn)[0]
            for fn in os.listdir(advanced_settings.LOAD_DATA_DIR)
        ],
        description="Load data file:",
        disabled=False,
    )
    display(input_collector["load_data_file"])

    def get_generator_options():
        year = input_collector["year"].value
        gen_data_file = advanced_settings.YEARLY_DATA_CACHE / f"gen_data_{year}.parquet"
        gen_regions = [input_collector["generator_region"].value]
        gen_options = import_data.get_generator_options(gen_data_file, gen_regions)
        return gen_options

    gen_options = get_generator_options()
    input_collector["generators"] = widgets.SelectMultiple(
        options=gen_options,
        value=tuple(gen_options[:4]),
        description="Generators:",
        disabled=False,
    )
    display(input_collector["generators"])

    def update_generator_options(change):
        if change["new"] != change["old"]:
            input_collector["generators"].options = get_generator_options()

    input_collector["year"].observe(update_generator_options)
    input_collector["generator_region"].observe(update_generator_options)

    display(
        HTML(
            """
        <h3>Contract parameters:</h3>
        """
        )
    )

    display(
        HTML(
            """
        <h4> - All contracts:</h4>
        """
        )
    )

    input_collector["contract_type"] = widgets.Dropdown(
        options=advanced_settings.CONTRACT_TYPES,
        value=advanced_settings.CONTRACT_TYPES[0],
        disabled=False,
    )

    button_setup(
        input_collector["contract_type"],
        "This determines the type of PPA to be modelled for your selected load and generation profiles.",
        "Contract type:",
    )

    input_collector["firming_contract_type"] = widgets.Dropdown(
        options=advanced_settings.FIRMING_CONTRACT_TYPES,
        value=advanced_settings.FIRMING_CONTRACT_TYPES[0],
        disabled=False,
    )
    button_setup(
        input_collector["firming_contract_type"],
        "This determines how residual load that is not covered under the PPA is purchased.",
        "Firming contract type:",
    )

    input_collector["settlement_period"] = widgets.Dropdown(
        options=advanced_settings.SETTLEMENT_PERIODS,
        value=advanced_settings.SETTLEMENT_PERIODS[0],
        disabled=False,
    )
    button_setup(
        input_collector["settlement_period"],
        "The settlement period refers to how frequently your bills are settled. For example, this can impact the outcomes for contracts with volume undersupply penalties as the volumes are calculated at the settlement period.",
        "Settlement period:",
    )

    input_collector["contract_amount"] = widgets.BoundedFloatText(
        value=100.0,
        min=0,
        max=1000.0,
    )
    button_setup(
        input_collector["contract_amount"],
        "For 24/7 contracts: this value sets the minimum guaranteed average hourly match to reach under the contract, sometimes known as the CFE score. For all other contracts this value sets the percentage of the total load volume you wish to contract through the PPA. For example, you may wish to contract for 110% of the annual load volume.",
        "Contract amount (%):",
    )

    input_collector["strike_price"] = widgets.FloatText(
        value=100.0,
    )
    button_setup(
        input_collector["strike_price"],
        "This is the contract price, around which settlements occur. If no floor price is set, this is how much each MWh of contracted energy costs.",
        "Strike price ($/MW/h):",
    )

    input_collector["lgc_buy_price"] = widgets.FloatText(
        value=35.0,
    )
    button_setup(
        input_collector["lgc_buy_price"],
        "This is the price for any extra LGCs purchased outside of the contract to meet the desired total volume.",
        "LGC buy price ($/MW/h):",
    )

    input_collector["lgc_sell_price"] = widgets.FloatText(
        value=20.0,
    )
    button_setup(
        input_collector["lgc_sell_price"],
        "This is the price at which any LGCs in excess of the desired total volume are sold by the buyer.",
        "LGC sell price ($/MW/h):",
    )

    input_collector["shortfall_penalty"] = widgets.FloatText(
        value=25.0,
    )
    button_setup(
        input_collector["shortfall_penalty"],
        "This is the penalty paid by the seller to the buyer for any contracted energy that is not delivered by the hybrid portfolio.",
        "Short fall penalty ($/MW/h):",
    )

    input_collector["floor_price"] = widgets.FloatText(
        value=0.0,
    )
    button_setup(
        input_collector["floor_price"],
        "This value sets a floor price for the contract, meaning that the wholesale price for the purposes of contract settlement has a minimum value of $0/MWh.",
        "Floor price ($/MW/h):",
    )

    input_collector["excess_price"] = widgets.FloatText(
        value=65.0,
    )
    button_setup(
        input_collector["excess_price"],
        "This sets the price at which excess energy (contracted energy above the load in a given interval) is sold by the buyer to earn on-sell revenue.",
        "Excess price ($/MW/h):",
    )

    input_collector["indexation"] = widgets.BoundedFloatText(
        value=1.0,
        min=0,
        max=100,
    )
    button_setup(
        input_collector["indexation"],
        "Sets the percentage indexation applied to the contract strike price in each index period (defined below).",
        "Indexation (%):",
    )

    input_collector["index_period"] = widgets.Dropdown(
        options=advanced_settings.INDEX_PERIODS,
        value=advanced_settings.INDEX_PERIODS[0],
        disabled=False,
    )
    button_setup(
        input_collector["index_period"],
        "This determines how often the strike price is indexed at the indexation rate.",
        "Index period:",
    )

    display(
        HTML(
            """
        <h4> - Shaped and baseload contracts only:</h4>
        """
        )
    )

    input_collector["redefine_period"] = widgets.Dropdown(
        options=advanced_settings.REDEFINE_PERIODS,
        value=advanced_settings.REDEFINE_PERIODS[2],
        disabled=False,
    )
    button_setup(
        input_collector["redefine_period"],
        "For Shaped and Baseload contract types, the contract shape or baseload amount can be set monthly, quarterly or yearly.",
        "Redefine period:",
    )

    display(
        HTML(
            """
        <h4> - Shaped contracts only:</h4>
        """
        )
    )

    input_collector["matching_percentile"] = widgets.BoundedFloatText(
        value=90.0,
        min=0,
        max=100,
    )
    button_setup(
        input_collector["matching_percentile"],
        "This sets the percentile value to take of each generator profile to create the optimised hybrid profile for Shaped contracts.",
        "Matching percentile:",
    )

    display(
        HTML(
            """
        <h4> - Partial Wholesale exposure only:</h4>
        """
        )
    )

    input_collector["exposure_upper_bound"] = widgets.FloatText(
        value=300.0,
    )
    button_setup(
        input_collector["exposure_upper_bound"],
        "If the firming contract type chosen is Partial Wholesale Exposure, this value sets an upper bound on the wholesale price (assuming that the buyer purchases caps or another hedging product to achieve this).",
        "Exposure upper bound ($/MW/h):",
    )

    input_collector["exposure_lower_bound"] = widgets.FloatText(
        value=20.0,
    )
    button_setup(
        input_collector["exposure_lower_bound"],
        "If the firming contract type chosen is Partial Wholesale Exposure, this value sets a lower bound on the wholesale price. This could reflect a partially exposed retail agreement.",
        "Exposure lower bound ($/MW/h):",
    )

    display(
        HTML(
            """
        <h4> - Analysis paramters:</h4>
        """
        )
    )

    input_collector["time_series_interval"] = widgets.Dropdown(
        options=advanced_settings.TIME_SERIES_INTERVALS,
        value=advanced_settings.TIME_SERIES_INTERVALS[0],
        description="Time series interval:",
        disabled=False,
    )
    display(input_collector["time_series_interval"])

    input_collector["generator_data_set"] = widgets.Dropdown(
        options=advanced_settings.GEN_COST_DATA.keys(),
        disabled=False,
    )
    button_setup(
        input_collector["generator_data_set"],
        "This defines the set of GenCost (or user supplied) data to use for the calculation of generator LCOEs.",
        "Generator data set:",
    )

    display(out)

    return input_collector


def get_unit_capacity(unit: str) -> float:
    """Returns the registered capacity of a generation unit in watts.

    Args:
        unit (str): The identifier of the unit, formatted as '<DUID>:<full_name>'.

    Returns:
        float: The registered capacity of the unit in watts (MW * 1000).

    The function extracts the DUID from the input string, queries the static table
    for the registered generation capacity in megawatts, and converts it to watts
    by multiplying by 1000.
    """
    duid = unit.split(":")[0]
    registered_capacity = static_table(
        table_name="Generators and Scheduled Loads",
        raw_data_location=advanced_settings.RAW_DATA_CACHE,
        select_columns=["DUID", "Reg Cap generation (MW)"],
        filter_cols=["DUID"],
        filter_values=[(duid,)],
    )["Reg Cap generation (MW)"].values[0]
    return float(registered_capacity) * 1000


def add_editor_for_generator(generator_data_editor, generator, input_collector):
    with generator_data_editor["out"]:
        generator_data_set_name = input_collector["generator_data_set"].value
        generator_data_set = advanced_settings.GEN_COST_DATA[generator_data_set_name]
        for generator_type in generator_data_set.keys():
            if generator_type.upper() in generator:
                generator_data_editor[f"{generator}"] = {}

                generator_data_editor[f"{generator}"]["label"] = HTML(
                    f"""
                    <h5>{generator}:</h5>
                    """
                )

                generator_data_editor[f"{generator}"]["Fixed O&M ($/kW)"] = (
                    widgets.FloatText(
                        value=generator_data_set[generator_type]["Fixed O&M ($/kW)"],
                        description="Fixed O&M ($/kW)",
                    )
                )

                generator_data_editor[f"{generator}"]["Variable O&M ($/kWh)"] = (
                    widgets.FloatText(
                        value=generator_data_set[generator_type][
                            "Variable O&M ($/kWh)"
                        ],
                        description="Variable O&M ($/kWh)",
                    )
                )

                generator_data_editor[f"{generator}"]["Capital ($/kW)"] = (
                    widgets.FloatText(
                        value=generator_data_set[generator_type]["Capital ($/kW)"],
                        description="Capital ($/kW)",
                    )
                )

                generator_data_editor[f"{generator}"]["Capacity Factor"] = (
                    widgets.FloatText(
                        value=generator_data_set[generator_type]["Capacity Factor"],
                        description="Capacity Factor",
                    )
                )

                generator_data_editor[f"{generator}"]["Construction Time (years)"] = (
                    widgets.FloatText(
                        value=generator_data_set[generator_type][
                            "Construction Time (years)"
                        ],
                        description="Construction Time (years)",
                    )
                )

                generator_data_editor[f"{generator}"]["Economic Life (years)"] = (
                    widgets.FloatText(
                        value=generator_data_set[generator_type][
                            "Economic Life (years)"
                        ],
                        description="Economic Life (years)",
                    )
                )

                capacity = get_unit_capacity(generator)

                generator_data_editor[f"{generator}"]["Nameplate Capacity (kW)"] = (
                    widgets.FloatText(
                        value=capacity,
                        description="Nameplate Capacity (kW)",
                    )
                )


def remove_editor_for_generator(generator_data_editor, generator):
    with generator_data_editor["out"]:
        for component in generator_data_editor[generator].keys():
            generator_data_editor[generator][component].close()
            del generator_data_editor[generator][component]
    del generator_data_editor[generator]


def update_generator_data_editor(generator_data_editor, input_collector, change=None):
    if change is None:
        for generator in input_collector["generators"].value:
            add_editor_for_generator(generator_data_editor, generator, input_collector)
    else:
        if isinstance(change["new"], str):
            change_new = [change["new"]]
        else:
            change_new = change["new"]
        if isinstance(change["old"], str):
            change_old = [change["old"]]
        else:
            change_old = change["old"]
        for generator in change_new:
            if generator not in change_old:
                add_editor_for_generator(
                    generator_data_editor, generator, input_collector
                )
        for generator in change_old:
            if generator not in change_new:
                remove_editor_for_generator(generator_data_editor, generator)
                pass

    generator_data_editor["out"].clear_output()
    with generator_data_editor["out"]:
        for key, value in generator_data_editor.items():
            if key != "out":
                for k, v in value.items():
                    display(v)

    display(generator_data_editor["out"])
    return generator_data_editor


def launch_generator_data_editor(input_collector):
    generator_data_editor = {}
    generator_data_editor["out"] = widgets.Output()
    generator_data_editor = update_generator_data_editor(
        generator_data_editor, input_collector
    )
    input_collector["generators"].observe(
        functools.partial(
            update_generator_data_editor, generator_data_editor, input_collector
        ),
        names="value",
    )
    return generator_data_editor


def launch_battery_input_collector():
    display(
        HTML(
            """
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 70ex !important; }
        .widget-dropdown select { min-width: 70ex !important; }
        .widget-floattext input { min-width: 70ex !important; }
        </style>
        """
        )
    )

    display(
        HTML(
            """
        <h3>Battery inputs:</h3>
        """
        )
    )

    battery_input_collector = {}

    battery_input_collector["rated_power_capacity"] = widgets.FloatText(
        value=1.0,
    )
    button_setup(
        battery_input_collector["rated_power_capacity"],
        "Set this to the desired battery power capacity to be modelled. This value is also used to set the maximum charge and discharge rate for the modelled battery.",
        "Rated power capacity (MW):",
    )

    battery_input_collector["size_in_mwh"] = widgets.FloatText(
        value=2.0,
    )
    button_setup(
        battery_input_collector["size_in_mwh"],
        "This should be equal to the storage duration in hours multiplied by the rated power capacity of the battery you wish to model, and sets the storage capacity of the battery.",
        "Battery size (MWh):",
    )

    return battery_input_collector


def launch_flex_input_collector():
    display(
        HTML(
            """
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 70ex !important; }
        .widget-dropdown select { min-width: 70ex !important; }
        .widget-floattext input { min-width: 70ex !important; }
        </style>
        """
        )
    )

    display(
        HTML(
            """
        <h3>Load flexibility inputs:</h3>
        """
        )
    )

    flex_input_collector = {}

    flex_input_collector["base_load_quantile"] = widgets.BoundedFloatText(
        value=0.8,
        min=0,
        max=1.0,
    )
    button_setup(
        flex_input_collector["base_load_quantile"],
        "This determines how much of the load will be allowed to shift in each day by setting a daily inflexible load profile based on the quantile input here. For example, a value of 0.5 here sets the median daily load profile as inflexible, with any load exceeding the median value in each interval allowed to shift within the day. Lower values = more flexible load. Run help(load_flex.daily_load_shifting) for details.",
        "Base load quantile:",
    )

    flex_input_collector["raise_price"] = widgets.FloatText(
        value=0.0,
    )
    button_setup(
        flex_input_collector["raise_price"],
        "This is a penalty value applied to any load increased above its original value in a given interval when optimising load flex. High input values will heavily restrict flexibility.",
        "Raise price ($/MWh):",
    )

    flex_input_collector["lower_price"] = widgets.FloatText(
        value=0.0,
    )
    button_setup(
        flex_input_collector["lower_price"],
        "This is a penalty value applied to any load decreased below its original value in a given interval when optimising load flex. High input values will heavily restrict flexibility.",
        "Lower price ($/MWh):",
    )

    flex_input_collector["ramp_up"] = widgets.FloatText(
        value=0.01,
    )
    button_setup(
        flex_input_collector["ramp_up"],
        "The penalty applied to the positive difference between load energy at time <i>t</i> and time <i>t-1</i>.",
        "Ramp up penalty ($/MWh):",
    )

    flex_input_collector["ramp_down"] = widgets.FloatText(
        value=0.01,
    )
    button_setup(
        flex_input_collector["ramp_down"],
        "The penalty applied to the negative difference between load energy at time <i>t</i> and time <i>t-1</i>.",
        "Ramp down penalty ($/MWh):",
    )

    return flex_input_collector


# Tariffs: Network tariff selection and extra charges collected to create retail
# tariff.
def tariff_options_collector(input_collector):
    display(
        HTML(
            """
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 70ex !important; }
        .widget-dropdown select { min-width: 70ex !important; }
        .widget-floattext input { min-width: 70ex !important; }
        </style>
        """
        )
    )

    display(
        HTML(
            """
        <h3>Network tariff selection:</h3>
        """
        )
    )

    tariff_collector = {}

    def get_tariff_options(input_collector):
        all_tariffs = helper_functions.read_json_file(
            advanced_settings.COMMERCIAL_TARIFFS_FN
        )
        all_tariffs = all_tariffs[0]["Tariffs"]

        tariff_options = []
        for i, tariff in enumerate(all_tariffs):
            if "CustomerType" in tariff:
                if tariff["CustomerType"] != "Residential":
                    # create the widgets
                    if tariff["State"] == input_collector["load_region"].value[:-1]:
                        tariff_options.append(tariff["Name"])
        return tariff_options

    tariff_options = get_tariff_options(input_collector)

    tariff_collector["tariff_name"] = widgets.Dropdown(
        options=tariff_options,
        value=tariff_options[0],
        description="Tariff name:",
        disabled=False,
    )
    display(tariff_collector["tariff_name"])

    def update_tariff_options(change):
        if change["new"] != change["old"]:
            tariff_collector["tariff_name"].options = get_tariff_options(
                input_collector
            )

    input_collector["load_region"].observe(update_tariff_options)

    return tariff_collector


def launch_extra_charges_collector():
    display(
        HTML(
            """
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 70ex !important; }
        .widget-dropdown select { min-width: 70ex !important; }
        .widget-floattext input { min-width: 70ex !important; }
        </style>
        """
        )
    )

    display(
        HTML(
            """
        <h3>Other Commercial Charges:</h3>
        """
        )
    )

    extra_charges_collector = {}

    display(
        HTML(
            """
        <h4>Energy Charges:</h4>
        """
        )
    )

    extra_charges_collector["peak_rate"] = widgets.FloatText(
        value=0.06, description="Peak rate ($/kWh):"
    )
    display(extra_charges_collector["peak_rate"])

    extra_charges_collector["shoulder_rate"] = widgets.FloatText(
        value=0.06, description="Shoulder rate ($/kWh):"
    )
    display(extra_charges_collector["shoulder_rate"])

    extra_charges_collector["off_peak_rate"] = widgets.FloatText(
        value=0.04, description="Off-Peak rate ($/kWh):"
    )
    display(extra_charges_collector["off_peak_rate"])

    extra_charges_collector["retailer_demand_charge"] = widgets.FloatText(
        value=0.00, description="Retailer demand charge ($/kVA/day):"
    )
    display(extra_charges_collector["retailer_demand_charge"])

    display(
        HTML(
            """
        <h4>Metering Charges:</h4>
        """
        )
    )

    extra_charges_collector["meter_provider_charge"] = widgets.FloatText(
        value=2.00, description="Meter Provider/Data Agent Charges ($/Day):"
    )
    display(extra_charges_collector["meter_provider_charge"])

    extra_charges_collector["other_meter_charge"] = widgets.FloatText(
        value=6.00, description="Other Meter Charges ($/Day):"
    )
    display(extra_charges_collector["other_meter_charge"])

    display(
        HTML(
            """
        <h4>Environmental Charges:</h4>
        """
        )
    )

    extra_charges_collector["lrec_charge"] = widgets.FloatText(
        value=0.008, description="LREC Charge ($/kWh):"
    )
    display(extra_charges_collector["lrec_charge"])

    extra_charges_collector["srec_charge"] = widgets.FloatText(
        value=0.004, description="SREC Charge ($/kWh):"
    )
    display(extra_charges_collector["srec_charge"])

    extra_charges_collector["state_env_charge"] = widgets.FloatText(
        value=0.002, description="State Environment Charge ($/kWh):"
    )
    display(extra_charges_collector["state_env_charge"])

    display(
        HTML(
            """
        <h4>AEMO Market Charges:</h4>
        """
        )
    )

    extra_charges_collector["participant_charge"] = widgets.FloatText(
        value=0.00036, description="AEMO Participant Charge ($/kWh):"
    )
    display(extra_charges_collector["participant_charge"])

    extra_charges_collector["ancillary_services_charge"] = widgets.FloatText(
        value=0.00018, description="AEMO Ancillary Services Charge ($/kWh):"
    )
    display(extra_charges_collector["ancillary_services_charge"])

    display(
        HTML(
            """
        <h4>Other Variable Charges:</h4>
        """
        )
    )

    extra_charges_collector["other_charge_one"] = widgets.FloatText(
        value=0.0, description="Other Variable Charge 1 ($/kWh):"
    )
    display(extra_charges_collector["other_charge_one"])

    extra_charges_collector["other_charge_two"] = widgets.FloatText(
        value=0.0, description="Other Variable Charge 2 ($/kWh):"
    )
    display(extra_charges_collector["other_charge_two"])

    extra_charges_collector["other_charge_three"] = widgets.FloatText(
        value=0.0, description="Other Variable Charge 3 ($/kWh):"
    )
    display(extra_charges_collector["other_charge_three"])

    display(
        HTML(
            """
        <h4>Other Fixed Charges:</h4>
        """
        )
    )

    extra_charges_collector["total_gst"] = widgets.FloatText(
        value=0.0, description="Total GST ($/Bill):"
    )
    display(extra_charges_collector["total_gst"])

    extra_charges_collector["other_fixed_charge_one"] = widgets.FloatText(
        value=0.0, description="Other Fixed Charge 1 ($/Bill):"
    )
    display(extra_charges_collector["other_fixed_charge_one"])

    extra_charges_collector["other_fixed_charge_two"] = widgets.FloatText(
        value=0.0, description="Other Fixed Charge 2 ($/Bill):"
    )
    display(extra_charges_collector["other_fixed_charge_two"])

    return extra_charges_collector


def collect_and_combine_data(
    input_collector: dict, tariff_collector: dict, extra_charges_collector: dict
) -> pd.DataFrame:
    # This function is currently only usable in the interface.ipynb notebook
    # as it relies on the various collector nested dictionary and widget
    # structures.

    # ----------------------------- Unpack user input ------------------------------
    year_to_load_from_cache = input_collector["year"].value
    year_to_load = int(year_to_load_from_cache)
    GENERATOR_REGION = input_collector["generator_region"].value
    LOAD_REGION = input_collector["load_region"].value
    generators = list(input_collector["generators"].value)

    # ------------------- Get Load Data --------------------
    # if using preset data, use these hard coded values:
    LOAD_DATA_DIR = "data_caches/c_and_i_customer_loads"
    load_filename = input_collector["load_data_file"].value + ".csv"
    filepath = LOAD_DATA_DIR + "/" + load_filename
    LOAD_DATETIME_COL_NAME = "TS"
    LOAD_COL_NAME = "Load"
    DAY_FIRST = True

    # Units are definitely a question.
    load_data, start_date, end_date = import_data.get_load_data(
        filepath, LOAD_DATETIME_COL_NAME, LOAD_COL_NAME, DAY_FIRST
    )
    load_data = load_data[
        (load_data.index >= f"{year_to_load}-01-01 00:00:00")
        & (load_data.index < f"{year_to_load+1}-01-01 00:00:00")
    ]

    # ----------------------------- Get Generation Data ----------------------------
    gen_data_file = (
        advanced_settings.YEARLY_DATA_CACHE
        / f"gen_data_{year_to_load_from_cache}.parquet"
    )
    gen_data = import_data.get_preprocessed_gen_data(gen_data_file, [GENERATOR_REGION])
    gen_data = gen_data[generators]

    # --------------------------- Get Emissions Data -------------------------------
    emissions_data_file = (
        advanced_settings.YEARLY_DATA_CACHE
        / f"emissions_data_{year_to_load_from_cache}.parquet"
    )
    emissions_intensity = import_data.get_preprocessed_avg_intensity_emissions_data(
        emissions_data_file, LOAD_REGION
    )

    # ------------------------ Get Wholesale Price Data ----------------------------
    price_data_file = (
        advanced_settings.YEARLY_DATA_CACHE
        / f"price_data_{year_to_load_from_cache}.parquet"
    )
    price_data = import_data.get_preprocessed_price_data(price_data_file, LOAD_REGION)

    combined_data = pd.concat(
        [load_data, gen_data, price_data, emissions_intensity], axis="columns"
    )

    FIRMING_CONTRACT_TYPE = input_collector["firming_contract_type"].value
    EXPOSURE_BOUND_UPPER = input_collector["exposure_upper_bound"].value
    EXPOSURE_BOUND_LOWER = input_collector["exposure_lower_bound"].value
    RETAIL_TARIFF_DETAILS = {}

    if input_collector["firming_contract_type"].value == "Retail":
        selected_tariff_name = tariff_collector["tariff_name"].value
        all_tariffs = helper_functions.read_json_file(
            advanced_settings.COMMERCIAL_TARIFFS_FN
        )
        all_tariffs = all_tariffs[0]["Tariffs"]
        for tariff in all_tariffs:
            if tariff["Name"] == selected_tariff_name:
                selected_tariff = copy.deepcopy(tariff)

        extra_charges = helper_functions.format_other_charges(extra_charges_collector)
        RETAIL_TARIFF_DETAILS = tariffs.add_other_charges_to_tariff(
            selected_tariff, extra_charges
        )

    # Add the firming details:
    combined_data = firming_contracts.choose_firming_type(
        FIRMING_CONTRACT_TYPE,
        combined_data,
        EXPOSURE_BOUND_UPPER,
        EXPOSURE_BOUND_LOWER,
        RETAIL_TARIFF_DETAILS,
    )

    combined_data = combined_data.dropna(how="any", axis="rows")

    return combined_data

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import saturation_curve_from_points as sim
import os
import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output
save_plots = False

if save_plots:
    folder_plots = 'plots_sat_func'
    os.makedirs(f'{folder_plots}', exist_ok=True)

def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)

def watts_to_photon_number(watts, omega):
    return watts / (hbar * omega)

def model_function(x, a, b, c):
    return a * (1 / (1 + (x / b))) + c

def model_function_no_c(x, a, b):
    return a * (1 / (1 + (x / b)))

def linear_model_function(x, a, b):
    return a * x + b

def find_decay_start_x(power_input, gain_adjusted, offset=0):
    # Calculate the difference in gain_adjusted
    gain_diff = np.diff(gain_adjusted)

    # Find the index of the first negative difference
    decay_start_index = np.where(gain_diff < 0)[0][0]  # Get the first negative diff index

    # Return the corresponding x value
    return power_input[decay_start_index + offset]

# omega = 6.0035e9  # Frequency in Hz
omega1 = np.mean([6.06239e9, 6.0632863e9])
hbar = 1.054571817e-34  # Reduced Planck's constant

def return_params(gain):
    # From simulated daatahis
    p1dbout = 18.4
    p1dbin = p1dbout + 1 - gain
    power_in = np.linspace(-30, 15, 5000)
    power_input_dbm, power_output_dbm = sim.create_amplifier_response_curve(gain, p1dbin, power_in)

    # Convert both to watts
    power_input_watts = dbm_to_watts(power_input_dbm)

    # Gain stage
    power_gain_db = power_output_dbm - power_input_dbm

    if save_plots:
        # PLOT 1 - Power input vs power output RAW
        # Plotting with photon number axes
        plt.plot(power_input_dbm, power_output_dbm, 'b-', label='Data')
        plt.xlabel('Input Power (dBm)')
        plt.ylabel('Output Power (dBm)')
        plt.title('Raw Saturation Curve Log-Log - Experimental Data')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{folder_plots}/function_from_spec_sheet_1.png')
        plt.close()

        # PLOT 2 - Power input (dBm) VS Power Gain (dB)
        plt.plot(power_input_dbm, power_gain_db, 'b-', label='Data')
        plt.xlabel('Input Power (dBm)')
        plt.ylabel('Output Gain (dB)')
        plt.title('Gain Curve Log-Log - Transformed Data')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{folder_plots}/function_from_spec_sheet_2.png')
        plt.close()

        # PLOT 3 - Power input (Watt) Vs Power Gain (dB)
        plt.plot(power_input_watts, power_gain_db, 'b-', label='Data')
        plt.xlabel('Input Power (Watts)')
        plt.ylabel('Output Gain (dB)')
        plt.title('Gain Curve SemilogX - Transformed Data')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{folder_plots}/function_from_spec_sheet_2.png')
        plt.close()

        # PLOT 4 - Power input (Watt) Vs Power Gain (Linear Scale)
        # Normally power dB to linear is /10, but since we have to square root it is /20
        linear_gain_adjusted = 10 ** (power_gain_db / 20)
        plt.figure(figsize=(10, 7))
        plt.plot(power_input_watts, linear_gain_adjusted, 'b-', label='Data')
        plt.xlabel('Input Power (Watts)')
        plt.ylabel('Output Gain (Linear Scale) - 10**(G0)/20')
        plt.title('Gain Curve Linear - Transformed Data')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_plots}/function_from_spec_sheet_3.png')
        plt.close()

    else:
        pass
    
    linear_gain_adjusted = 10 ** (power_gain_db / 20)
    
    # Now, do the fitting for Plot 4
    USE_C = True

    x_value_when_decay_starts = find_decay_start_x(power_input_watts, linear_gain_adjusted)
    filtered_power_input_watts = power_input_watts[power_input_watts >= x_value_when_decay_starts]
    filtered_linear_gain_adjusted = linear_gain_adjusted[power_input_watts >= x_value_when_decay_starts]

    flat_line_value = linear_gain_adjusted[0]

    def model_function_with_flat_c(x, a, b, c, x0):
        return np.where(x <= x0, flat_line_value, model_function(x, a, b, c))

    def model_function_with_flat(x, a, b, x0):
        return np.where(x <= x0, flat_line_value, model_function_no_c(x, a, b))

    if USE_C:
        params, covariance = curve_fit(model_function, filtered_power_input_watts, filtered_linear_gain_adjusted,
                                    method='lm', maxfev=100000)

        # params will contain the fitted values for a and b
        a_fitted, b_fitted, c_fitted = params
        x_values = np.linspace(min(power_input_watts), max(power_input_watts), 500)
        y_values_fitted = model_function(x_values, a_fitted, b_fitted, c_fitted)
        offset_x_start = find_decay_start_x(power_input_watts, linear_gain_adjusted, offset=75)
        y_values_fitted_new = model_function_with_flat_c(x_values, a_fitted, b_fitted, c_fitted, offset_x_start)

        if save_plots:
            plt.plot(x_values, y_values_fitted_new, 'r--', label='Fit')
            plt.plot(filtered_power_input_watts, filtered_linear_gain_adjusted, 'b-', label='Data')
            plt.xlabel('Input Power (Watts)')
            plt.ylabel('Output Gain (Linear Scale) - 10**(G0)/20')
            plt.title('Gain Curve Linear - Transformed Data')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{folder_plots}/function_from_spec_sheet_4.png')
            plt.close()
        else:
            pass
        
    else:
        params, covariance = curve_fit(model_function_no_c, filtered_power_input_watts, filtered_linear_gain_adjusted,
                                    method='lm', maxfev=100000)

        # params will contain the fitted values for a and b
        a_fitted, b_fitted = params
        x_values = np.linspace(min(power_input_watts), max(power_input_watts), 500)
        y_values_fitted = model_function_no_c(x_values, a_fitted, b_fitted)
        offset_x_start = find_decay_start_x(power_input_watts, linear_gain_adjusted, offset=75)
        y_values_fitted_new = model_function_with_flat(x_values, a_fitted, b_fitted, offset_x_start)

    return params[0], params[1], params[2], flat_line_value, offset_x_start

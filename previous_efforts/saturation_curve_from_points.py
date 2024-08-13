import numpy as np
import matplotlib.pyplot as plt

# Function to calculate a quadratic function from 3 points
def find_quadratic_from_points(p1, p2, p3):
    """
    Find a quadratic equation (a*x^2 + b*x + c) given 3 points (x, y).
    Points are tuples (x, y).
    """
    # Create system of equations to solve for a, b, and c
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Set up the matrices
    A = np.array([[x1 ** 2, x1, 1],
                  [x2 ** 2, x2, 1],
                  [x3 ** 2, x3, 1]])

    B = np.array([y1, y2, y3])

    # Solve for a, b, and c
    a, b, c = np.linalg.solve(A, B)

    # Return the quadratic function as a lambda function
    return lambda x: a * x ** 2 + b * x + c, (a, b, c)


def create_amplifier_response_curve(gain, p1dbin, power_in):
    """
    Create a curve that represents the output power of an amplifier as a function of the input power.
    :param gain:
    :param p1dbin:
    :param power_in:
    :return:
    """
    # Linear output power values
    power_out = power_in + gain

    # Find the output power at 1 dB compression point
    op1db = power_out[abs(power_in - p1dbin) <= .01] - 1
    op1db = op1db[0]

    # Define the quadratic and flat region transition points
    first_x = p1dbin - 2
    first_y = op1db - 1
    x_quad = np.array([first_x, p1dbin, p1dbin + 2])
    y_quad = np.array([first_y, op1db, op1db + .33])

    # Linear region data
    power_in_linear_regime = power_in[power_in < first_x]
    power_out_linear_regime = power_out[power_in < first_x]

    # Find the quadratic function that goes through these 3 points
    quadratic_func, _ = find_quadratic_from_points((x_quad[0], y_quad[0]), (x_quad[1], y_quad[1]),
                                                   (x_quad[2], y_quad[2]))

    # Generate the quadratic and flat region data
    quadratic_x = np.linspace(x_quad[0], x_quad[2], 1000)
    quadratic_y = quadratic_func(quadratic_x)

    power_max = np.max(power_in)
    # flat_x = np.linspace(x_quad[2], x_quad[2] + 5, 1000)
    flat_x = np.linspace(x_quad[2], power_max, 1000)
    flat_y = np.ones(1000) * quadratic_y[-1]

    # Combine all the x and y data into single vectors
    combined_x = np.concatenate((power_in_linear_regime, quadratic_x, flat_x))
    combined_y = np.concatenate((power_out_linear_regime, quadratic_y, flat_y))

    return combined_x, combined_y


def main():
    
    # Amplifier parameters
    gain = 19.8
    p1dbout = 18.4
    p1dbin = p1dbout + 1 - gain

    power_in = np.linspace(-30, 25, 5000)

    cx, cy = create_amplifier_response_curve(gain, p1dbin, power_in)
    plt.plot(cx, cy)
    plt.savefig('plot_1.png')
    plt.close()

if __name__ == '__main__':
    main()